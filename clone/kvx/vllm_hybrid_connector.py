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
import re
from dataclasses import dataclass, field
from typing import Any

import torch

from . import hybrid_state as hs
from .hybrid_state import caches_from_taps, k_norm_weights, taps_from_feats, write_lin_state

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
        self.min_len = int(cfg.get_from_extra_config("min_len", 512))
        self.src_name = cfg.get_from_extra_config("src", None)
        self.maps_path = cfg.get_from_extra_config("maps", None)
        self.sinks = cfg.get_from_extra_config("sinks", None)  # override the file's sink count (template rows)
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
        if n < self.min_len:
            return 0
        t = self.tail if tail is None else max(1, int(tail))
        return max(0, ((n - t) // self.block) * self.block)

    @staticmethod
    def _req_tail(request):
        """Per-request native tail: SamplingParams(extra_args={"kv_transfer_params": {"kvx_tail": N}}), e.g. the
        question length so the document is mapped and the question read natively. None -> the engine-wide tail."""
        p = getattr(request, "kv_transfer_params", None) or {}
        return p.get("kvx_tail") if isinstance(p, dict) else None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        m = self._n_mapped(len(request.prompt_token_ids or []), self._req_tail(request))
        self.__dict__.setdefault("_m_req", {})[request.request_id] = m
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
        # mmap load, then straight to the GPU in bf16 (load_maps_file's fp32 copy would need ~2x the 18 GB file in RAM)
        self.mf = torch.load(self.maps_path, map_location="cpu", mmap=True, weights_only=False)
        assert self.mf.get("format") == "kvx-hybrid-v1", self.mf.get("format")
        self.maps = {int(t): m for t, m in self.mf["maps"].items()}
        if self.mf["src"] != self.src_name or self.mf["tgt"] not in (self.tgt_name, self.tgt_name.rstrip("/").split("/")[-1]):
            logger.warning("kvx hybrid: maps file src/tgt %s/%s vs serving %s/%s", self.mf["src"], self.mf["tgt"],
                           self.src_name, self.tgt_name)
        self.maps = {t: dict(sel=m["sel"], W=m["W"].to(device, torch.bfloat16), b=m["b"].float().to(device))
                     for t, m in self.maps.items()}
        self.mf["maps"] = None  # drop the mmap references
        tcfg = AutoConfig.from_pretrained(self.tgt_name)
        tcfg = getattr(tcfg, "text_config", None) or tcfg
        # small target params straight from the checkpoint: conv1d, A_log, dt_bias, k_norm
        path = snapshot_download(self.tgt_name, local_files_only=True)
        import glob, os
        want = re.compile(r"layers\.\d+\.(linear_attn\.(conv1d\.(weight|bias)|A_log|dt_bias)|self_attn\.k_norm\.weight)$")
        sd = {}
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            with safe_open(f, "pt") as h:
                for k in h.keys():
                    if want.search(k):
                        sd[re.sub(r"^.*?(layers\.)", "model.layers.", k)] = h.get_tensor(k)
        self.params = hs.TargetParams.from_state_dict(tcfg, sd).to(device)
        self.knw = {l: w.to(device) for l, w in k_norm_weights(tcfg, sd).items()}
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
        logger.info("kvx hybrid connector ready: %d mapped target layers, attn block %d, conv dim_first=%s",
                    len(self.maps), self.block, self.dim_first)

    def start_load_kv(self, forward_context, **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        if not isinstance(meta, HybridMeta) or not meta.requests:
            return
        if not getattr(self, "_kv", None):
            self.register_kv_caches({n: l.kv_cache[0] if isinstance(l.kv_cache, (list, tuple)) and len(l.kv_cache) == 1
                                     else l.kv_cache for n, l in forward_context.no_compile_layers.items()
                                     if getattr(l, "kv_cache", None) is not None})
        device = next(t for t in self._kv.values() if isinstance(t, torch.Tensor)).device if any(
            isinstance(t, torch.Tensor) for t in self._kv.values()) else torch.device("cuda")
        if not self._ready:
            self._init_worker(device)
        from . import hybrid
        for r in meta.requests:
            ids = torch.tensor([r.token_ids], device=device)
            T = len(r.token_ids)
            feats = hybrid.source_features(self.Sdec, ids, drop_q=self.mf.get("drop_q", True))
            given = taps_from_feats(self.mf, self.maps, feats, T, self.knw, sinks=self.sinks)
            del feats
            states = caches_from_taps(self.params, self.kinds, given)
            del given
            for t, st in states.items():
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
