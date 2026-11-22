"""vLLM v1 KV connector for HYBRID targets (Qwen3.5 / Qwen3.8 GatedDeltaNet + full attention), kvx-hybrid-v1 maps.

    llm = LLM(model="Qwen/Qwen3.8-27B", enable_prefix_caching=False, kv_transfer_config=KVTransferConfig(
        kv_connector="HybridCrossModelConnector", kv_connector_module_path="kvx.vllm_hybrid_connector",
        kv_role="kv_both", kv_connector_extra_config={"src": "Qwen/Qwen3.5-4B", "maps": "/vol/runs/<id>/hybrid_maps.pt",
                                                      "tail": 256, "min_len": 512}))

A prompt of n >= min_len tokens gets m = align_down(n - tail, attention block) tokens reported as an external hit.
start_load_kv runs the source (HF) on prompt[:m], maps its taps to the target's taps (kvx-pairs' maps, kvx/hybrid.py
format), and writes what the target's own forward would have cached (kvx/hybrid_state.py):
  attention layers -> paged K (partial RoPE) / V blocks;  GatedDeltaNet layers -> conv_state + recurrent_state slot.
The target then prefills only the last n - m tokens natively, starting from those states.
Needs vLLM's hybrid memory allocator (SupportsHMA) and prefix caching OFF (one mamba state slot per request).
"""
import os
import re
from dataclasses import dataclass, field
from typing import Any

import torch

from . import hybrid_state as hs
from .hybrid_state import caches_from_taps, k_norm_weights, slice_for_rank, taps_from_feats, write_lin_state

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


# ---------------------------------------------------------------- vLLM connector

from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole, SupportsHMA)
from vllm.logger import init_logger  # noqa: E402

logger = init_logger("vllm.kvx_hybrid_connector")


@dataclass
class _HReq:
    token_ids: list
    block_ids: tuple  # per KV cache group


@dataclass
class HybridMeta(KVConnectorMetadata):
    requests: list = field(default_factory=list)


class HybridCrossModelConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        cfg = self._kv_transfer_config
        self.tail = int(cfg.get_from_extra_config("tail", 256))
        self.align = int(cfg.get_from_extra_config("align", 1))
        self.tail_frac = float(cfg.get_from_extra_config("tail_frac", 0.25))
        self.min_len = int(cfg.get_from_extra_config("min_len", 512))
        self.src_name = cfg.get_from_extra_config("src", None)
        self.maps_path = cfg.get_from_extra_config("maps", None)
        # xkv maps (amw's recipe; kvx-pairs' kvx/xkv_adapter.py): source = input_layernorm outputs, same target taps
        self.xkv_mapper = cfg.get_from_extra_config("xkv_mapper", None)
        self.xkv_attn = cfg.get_from_extra_config("xkv_attn", None)
        # first m target layers computed natively over the whole prompt (exact token identity; a labelled config,
        # cost ~ m/num_layers of a target prefill, not part of the native-tail cap)
        self.native_layers = int(cfg.get_from_extra_config("native_layers", 0))
        self.sinks = cfg.get_from_extra_config("sinks", None)  # override the file's sink count (template rows)
        # map_once (multi-turn, OneTriangle's "exact mapped cache"): a conversation's FIRST request maps all but its
        # native tail; every later request that starts with that mapped prefix maps exactly the same prefix (causal
        # and deterministic, so the same KV/state as reusing the cache) and the target computes everything after it
        # natively: later user turns, tool outputs and the agent's own replies never go through the map.
        self.map_once = bool(int(cfg.get_from_extra_config("map_once", 0)))
        self._prefixes = {}  # first-prompt length -> {hash(first prompt tokens): its mapped prefix length}
        self._mstats = dict(first_n=0, first_prompt=0, first_mapped=0, later_n=0, later_prompt=0, later_mapped=0)
        self.tgt_name = vllm_config.model_config.model
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("kvx hybrid connector: start vLLM with enable_prefix_caching=False (one state slot per request)")
        self.kv_cache_config = kv_cache_config
        self.groups = []  # per group: (kind, block_size, layer ids)
        if kv_cache_config is not None:
            for g in kv_cache_config.kv_cache_groups:
                spec = g.kv_cache_spec
                kind = "lin" if "Mamba" in type(spec).__name__ else "attn"
                self.groups.append((kind, int(spec.block_size), [int(_LAYER_RE.search(n).group(1)) for n in g.layer_names]))
        attn_bs = [bs for k, bs, _ in self.groups if k == "attn"]
        self.block = attn_bs[0] if attn_bs else vllm_config.cache_config.block_size
        self._need_load = {}
        self._m_req = {}  # request id -> mapped tokens decided in get_num_new_matched_tokens (per-request tail)
        self._ready = False
        self.stats = dict(requests=0, mapped_tokens=0)

    # ---------------- scheduler side
    def _n_mapped(self, n, tail=None):
        """Mapped prefix length. align=1 (default): exactly n - tail (attention KV may end inside a block; the GDN state
        covers exactly the first m tokens). Hybrid attention blocks are large (e.g. 528-576 tokens), so block alignment
        mapped NOTHING on prompts shorter than a block (short benchmarks ran fully native)."""
        if n < self.min_len:
            return 0
        t = self.tail if tail is None else max(1, int(tail))
        t = max(1, min(t, int(self.tail_frac * n)))  # bundle rule: native tail <= 25% of the prompt
        a = max(1, self.align)
        return max(0, ((n - t) // a) * a)

    @staticmethod
    def _req_tail(request):
        """Per-request native tail: SamplingParams(extra_args={"kv_transfer_params": {"kvx_tail": N}}), e.g. the
        question length so the document is mapped and the question read natively. None -> the engine-wide tail."""
        p = getattr(request, "kv_transfer_params", None) or {}
        return p.get("kvx_tail") if isinstance(p, dict) else None

    def _map_once(self, ids, tail):
        """Mapped prefix length under map_once. A later turn starts with its conversation's WHOLE first prompt: it maps
        exactly that first request's mapped prefix. Otherwise this is a first request: n - tail, remembered."""
        n = len(ids)
        for N in sorted(self._prefixes, reverse=True):  # N = a first prompt's length -> {hash(first prompt): m}
            if N <= n:
                L = self._prefixes[N].get(hash(tuple(ids[:N])))
                if L is not None:
                    st = self._mstats
                    st["later_n"] += 1
                    st["later_prompt"] += n
                    st["later_mapped"] += L
                    return L
        m = self._n_mapped(n, tail)
        if m > 0:
            self._prefixes.setdefault(n, {})[hash(tuple(ids))] = m
            if sum(len(v) for v in self._prefixes.values()) > 4096:  # bound memory: forget the shortest prompts
                self._prefixes.pop(min(self._prefixes))
        st = self._mstats
        st["first_n"] += 1
        st["first_prompt"] += n
        st["first_mapped"] += m
        if st["first_n"] % 10 == 0 or (st["later_n"] and st["later_n"] % 50 == 0):
            logger.info("kvx map_once: first requests %d mapped %.4f of their prompts; later requests %d mapped "
                        "%.4f; all %.4f", st["first_n"], st["first_mapped"] / max(1, st["first_prompt"]),
                        st["later_n"], st["later_mapped"] / max(1, st["later_prompt"]),
                        (st["first_mapped"] + st["later_mapped"]) / max(1, st["first_prompt"] + st["later_prompt"]))
        return m

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        mr = self.__dict__.setdefault("_m_req", {})
        if request.request_id in mr:  # the scheduler may ask again for a waiting request: decide once
            m = mr[request.request_id]
        elif self.map_once:
            m = self._map_once(list(request.prompt_token_ids or []), self._req_tail(request))
        else:
            m = self._n_mapped(len(request.prompt_token_ids or []), self._req_tail(request))
        mr[request.request_id] = m
        return max(0, m - num_computed_tokens), False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens > 0:
            self._need_load[request.request_id] = request

    def build_connector_meta(self, scheduler_output):
        meta = HybridMeta()
        for nr in scheduler_output.scheduled_new_reqs:
            if nr.req_id not in self._need_load:
                continue
            ids = list(nr.prompt_token_ids or [])
            m = self.__dict__.setdefault("_m_req", {}).pop(nr.req_id, None)
            m = self._n_mapped(len(ids)) if m is None else m
            meta.requests.append(_HReq(token_ids=ids[:m], block_ids=tuple(nr.block_ids)))
        self._need_load.clear()
        return meta

    def request_finished(self, request, block_ids):
        self.__dict__.setdefault("_m_req", {}).pop(request.request_id, None)
        return False, None

    def request_finished_all_groups(self, request, block_ids):
        self.__dict__.setdefault("_m_req", {}).pop(request.request_id, None)
        return False, None

    # ---------------- worker side
    def register_kv_caches(self, kv_caches):
        self._kv = {}
        for name, t in kv_caches.items():
            m = _LAYER_RE.search(name)
            if m:
                self._kv[int(m.group(1))] = t

    def _init_worker(self, device):
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from transformers import AutoConfig
        from . import hybrid
        from .vllm_connector import kv_layout
        self.kv_layout = kv_layout
        self.xm = None
        if self.xkv_mapper:
            from .xkv_adapter import load_xkv
            self.xm = load_xkv(self.xkv_mapper, self.xkv_attn, device=device)
            self.mf = dict(format="kvx-hybrid-v1", maps={}, tgt_taps=None, src=self.src_name, tgt=self.tgt_name)
            self.maps_path = None
        if self.maps_path is None and self.xm is None:
            raise ValueError("kvx hybrid connector: set extra_config 'maps' (kvx-hybrid-v1) or 'xkv_mapper' (+ 'xkv_attn')")
        # mmap load, then straight to the GPU in bf16 (load_maps_file's fp32 copy would need ~2x the 18 GB file in RAM)
        if self.maps_path:
            self.mf = torch.load(self.maps_path, map_location="cpu", mmap=True, weights_only=False)
            assert self.mf.get("format") == "kvx-hybrid-v1", self.mf.get("format")
            self.maps = {int(t): m for t, m in self.mf["maps"].items()}
            if self.mf["src"] != self.src_name or self.mf["tgt"] not in (self.tgt_name, self.tgt_name.rstrip("/").split("/")[-1]):
                logger.warning("kvx hybrid: maps file src/tgt %s/%s vs serving %s/%s", self.mf["src"], self.mf["tgt"],
                               self.src_name, self.tgt_name)
            # kept on CPU here: sharded per TP rank below (after the target params), then moved to the GPU
            opt = lambda x: None if x is None else x.float()
            self.maps = {t: dict(sel=m["sel"], W=m["W"].to(torch.bfloat16), b=m["b"].float(),
                                 rc=opt(m.get("rc")), mu=opt(m.get("mu"))) for t, m in self.maps.items()}
        else:
            self.maps = {}
        self.recolor = bool(int(cfg_recolor)) if (cfg_recolor := self._kv_transfer_config.get_from_extra_config(
            "recolor", None)) is not None else True
        self.mf["maps"] = None  # drop the mmap references
        tcfg = AutoConfig.from_pretrained(self.tgt_name)
        tcfg = getattr(tcfg, "text_config", None) or tcfg
        rope_ov = self._kv_transfer_config.get_from_extra_config("rope_parameters", None)
        if rope_ov:  # e.g. YaRN x4 for 1M: must match the hf_overrides vLLM got, or mapped keys get the wrong RoPE
            tcfg.rope_parameters = dict(rope_ov)
        # small target params straight from the checkpoint: conv1d, A_log, dt_bias, k_norm
        # vLLM may hand us an already resolved local snapshot dir instead of a repo id
        path = self.tgt_name if os.path.isdir(self.tgt_name) else snapshot_download(self.tgt_name, local_files_only=True)
        import glob
        want = re.compile(r"layers\.\d+\.(linear_attn\.(conv1d\.(weight|bias)|A_log|dt_bias)|self_attn\.k_norm\.weight)$")
        sd = {}
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            with safe_open(f, "pt") as h:
                for k in h.keys():
                    if want.search(k):
                        sd[re.sub(r"^.*?(layers\.)", "model.layers.", k)] = h.get_tensor(k)
        self.params = hs.TargetParams.from_state_dict(tcfg, sd).to(device)
        self.full_params = self.params
        self.prefix = None
        if self.native_layers > 0:
            m_ = self.native_layers
            keep = re.compile(r"(embed_tokens\.weight|layers\.(\d+)\.)")
            psd = {}
            for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
                with safe_open(f, "pt") as h:
                    for k in h.keys():
                        mm = keep.search(k)
                        if mm and "visual" not in k and (mm.group(2) is None or int(mm.group(2)) < m_):
                            psd[k] = h.get_tensor(k)
            self.prefix = hs.build_prefix_model(tcfg, psd, m_, device=device)
            del psd
        self.knw = {l: w.to(device) for l, w in k_norm_weights(tcfg, sd).items()}
        from .vllm_connector import _tp_rank_size
        self.tp_rank, self.tp_size = _tp_rank_size()
        self._sharded = False
        if self.maps and self.xm is None:
            if self.tp_size > 1:  # each rank keeps only its heads' map columns (1/tp of the 18 GB of maps)
                self.maps, taps_r, self.params = hs.shard_for_rank(
                    self.mf, self.maps, self.params, self.tp_rank, self.tp_size, int(tcfg.num_key_value_heads))
                self.mf["tgt_taps"] = taps_r
                self._sharded = True
            mv = lambda x: None if x is None else x.to(device)
            self.maps = {t: dict(sel=m["sel"], W=m["W"].to(device), b=m["b"].to(device), rc=mv(m.get("rc")),
                                 mu=mv(m.get("mu"))) for t, m in self.maps.items()}
        if self.xm is not None:
            self.kinds = {t: ("lin" if lt == "linear_attention" else "attn") for t, lt in enumerate(tcfg.layer_types)}
            self.tcfg = tcfg
        else:
            self.kinds = {t: x["kind"] for t, x in enumerate(self.mf["tgt_taps"])}
        self.S = hybrid.load(self.src_name).model.to(device)
        self.Sdec = hybrid.Dec(self.S)
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
        self.dim_first = bool(is_conv_state_dim_first())
        self.layer_group = {}
        for gi, (kind, bs, layers) in enumerate(self.groups):
            for l in layers:
                self.layer_group[l] = gi
        self._layout = None
        self._ready = True
        logger.info("kvx hybrid connector ready: %d mapped target layers, attn block %d, conv dim_first=%s, tp %d/%d",
                    len(self.maps), self.block, self.dim_first, self.tp_rank, self.tp_size)

    def start_load_kv(self, forward_context, **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        if not isinstance(meta, HybridMeta) or not meta.requests:
            return
        # vLLM 0.34 registers ONE raw allocation per layer; each layer's own bind_kv_cache unpacks it into views
        # (attention: the paged KV tensor; GatedDeltaNet: (conv_state, ssm_state)). Write through those views.
        views = {}
        for n, l in forward_context.no_compile_layers.items():
            m = _LAYER_RE.search(n)
            kvc = getattr(l, "kv_cache", None)
            if m is None or kvc is None:
                continue
            if isinstance(kvc, (list, tuple)) and len(kvc) == 1:  # per-virtual-engine wrapper
                kvc = kvc[0]
            views[int(m.group(1))] = kvc
        if views:
            self._kv = views
        device = next(t for t in self._kv.values() if isinstance(t, torch.Tensor)).device if any(
            isinstance(t, torch.Tensor) for t in self._kv.values()) else torch.device("cuda")
        if not self._ready:
            self._init_worker(device)
        from . import hybrid
        for r in meta.requests:
            ids = torch.tensor([r.token_ids], device=device)
            T = len(r.token_ids)
            if self.xm is not None:
                from .xkv_adapter import needed_layers, source_inputs
                xf, emb, emb_prev = source_inputs(self.Sdec, ids, needed_layers(self.xm))
                layers = sorted(self.kinds)
            else:
                feats = hybrid.source_features(self.Sdec, ids, drop_q=self.mf.get("drop_q", True))
                layers = sorted(self.maps)
            # one target layer at a time: taps -> state -> write -> free (all 64 layers' fp32 taps of a 27K-token
            # prompt at once ran an H200 out of memory)
            ntaps = hs.native_prefix_taps(self.prefix, ids) if self.prefix is not None else {}
            for t in layers:
                if t in ntaps:  # native first layers override the map
                    given = {t: ntaps[t]}
                elif self.xm is not None:
                    tp = hs.xkv_layer_taps(self.xm, xf, emb, emb_prev, t, self.kinds[t], self.tcfg)
                    given = {t: {k: v[0] for k, v in tp.items()}}
                else:
                    given = taps_from_feats(self.mf, {t: self.maps[t]}, feats, T, self.knw, sinks=self.sinks,
                                            recolor=self.recolor)
                # native-layer taps are full width: their states use the unsharded params, then get sliced
                st = caches_from_taps(self.full_params if t in ntaps else self.params, self.kinds, given)[t]
                del given
                if not self._sharded or t in ntaps or self.xm is not None:  # sharded maps give rank-local states
                    st = slice_for_rank(self.full_params if self._sharded else self.params, st, self.tp_rank, self.tp_size)
                cache = self._kv.get(t)
                if cache is None:
                    raise RuntimeError(f"kvx hybrid: target layer {t} has no cache")
                blocks = r.block_ids[self.layer_group[t]]
                if st[0] == "attn":
                    _, K, V = st
                    bs = self.groups[self.layer_group[t]][1]
                    slots = (torch.arange(bs)[None, :] + torch.tensor(blocks)[:, None] * bs).flatten()[:T].to(device)
                    bi, off = slots // bs, slots % bs
                    kv = cache[0] if isinstance(cache, (list, tuple)) else cache
                    if self._layout is None:
                        self._layout = self.kv_layout(kv, K.shape[1], K.shape[2], "")
                    from .vllm_connector import write_kv
                    write_kv(kv, bi, off, K.to(torch.bfloat16), V.to(torch.bfloat16), bs, self._layout)
                else:
                    _, conv, S = st
                    if not getattr(self, "_logged_lin", False):
                        def desc(x, d=0):
                            if torch.is_tensor(x):
                                return f"Tensor{tuple(x.shape)} {x.dtype} stride={x.stride()}"
                            if isinstance(x, (list, tuple)) and d < 3:
                                return f"{type(x).__name__}[" + ", ".join(desc(y, d + 1) for y in x) + "]"
                            return type(x).__name__
                        fc_layer = next((l for n, l in forward_context.no_compile_layers.items()
                                         if _LAYER_RE.search(n) and int(_LAYER_RE.search(n).group(1)) == t), None)
                        logger.info("kvx hybrid: lin layer %d registered cache %s | layer.kv_cache %s | blocks %s | "
                                    "conv %s S %s", t, desc(cache), desc(getattr(fc_layer, "kv_cache", None)),
                                    blocks[:4], tuple(conv.shape), tuple(S.shape))
                        self._logged_lin = True
                    if not (isinstance(cache, (list, tuple)) and len(cache) >= 2):
                        raise RuntimeError(f"kvx hybrid: unexpected GDN cache object for layer {t}: see the log line above")
                    conv_c, ssm_c = cache[0], cache[1]
                    write_lin_state(conv_c, ssm_c, int(blocks[0]), conv, S, self.dim_first)
            self.stats["requests"] += 1
            self.stats["mapped_tokens"] += T

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        return

    def wait_for_save(self):
        return
