"""vLLM v1 KV connector for Gemma-4 targets (and other attention-only models with per-layer head shapes).

    kv_transfer_config=KVTransferConfig(kv_connector="GemmaCrossModelConnector",
        kv_connector_module_path="kvx.vllm_gemma_connector", kv_role="kv_both",
        kv_connector_extra_config={"src": "google/gemma-4-E4B", "maps": "<run>/maps_core.pt",
                                   "sink_template": "<run>/sink_template.pt", "tail": 256})

Gemma 4 differs from the uniform-layer path in kvx/vllm_connector.py:
  - two attention kinds per model with different KV shapes (31B: sliding 16 x 256, window 1024; global 4 x 512) and
    their own RoPE (sliding: default theta 1e4; global: "proportional", partial 0.25, theta 1e6: frequencies over
    head_dim, the non-rotated angles have zero frequency), two vLLM KV cache groups (hybrid memory allocator);
  - the cache holds K = RoPE(k_norm(k)) and V = v_norm(v) (unit RMS, no weight); global layers have K = V projections
    (attention_k_eq_v), so both slots are written from their maps;
  - sliding layers keep only their window: the request's block list for that group has null blocks for evicted
    positions, so only the last `window` mapped tokens are written there.
Maps: modal_fit_eval's maps_k format (kvx/hybrid.py to_core_maps): per target layer {sel: source layers, W: rows =
those layers' K (or V) taps in sel order, b, H, D, layer}; source taps = k_norm / v_norm outputs (kvx/hybrid.py
source_features). Restore: K re-normalized with the target k_norm weight, V to unit RMS per head.
"""
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole, SupportsHMA)
from vllm.logger import init_logger

logger = init_logger("vllm.kvx_gemma_connector")
_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def gemma_rope_tables(rp, head_dim, T, device):
    """cos/sin [T, head_dim] for one Gemma-4 layer type (rope_parameters entry): default or proportional."""
    theta = float(rp.get("rope_theta", 10000.0))
    if rp.get("rope_type", "default") == "proportional":
        angles = int(head_dim * float(rp.get("partial_rotary_factor", 1.0))) // 2
        inv = 1.0 / theta ** (torch.arange(0, 2 * angles, 2, dtype=torch.float32, device=device) / head_dim)
        inv = torch.cat([inv, torch.zeros(head_dim // 2 - angles, device=device)])
    else:
        inv = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    f = torch.outer(torch.arange(T, device=device, dtype=torch.float32), inv)
    e = torch.cat([f, f], -1)
    return e.cos(), e.sin()


def _rope(x, cos, sin):  # x [T, H, D] float32
    d = x.shape[-1] // 2
    return x * cos[:, None] + torch.cat([-x[..., d:], x[..., :d]], -1) * sin[:, None]


def restore_k(k, w, eps=1e-6):
    """k [T, H, D] mapped pre-RoPE key -> w * u / rms(u), u = k / w (Gemma4RMSNorm multiplies by w itself); channels
    with |w| ~ 0 keep w*0 and do not enter the RMS."""
    keep = w.abs() > 1e-3 * w.abs().amax()
    u = torch.where(keep, k / torch.where(keep, w, torch.ones_like(w)), torch.zeros_like(k))
    ms = (u * u * keep).sum(-1, keepdim=True) / keep.sum().clamp(min=1)
    return w * u * torch.rsqrt(ms + eps)


def restore_v(v, eps=1e-6):
    return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)


@dataclass
class _GReq:
    token_ids: list
    block_ids: tuple  # per KV cache group


@dataclass
class GemmaMeta(KVConnectorMetadata):
    requests: list = field(default_factory=list)


class GemmaCrossModelConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        cfg = self._kv_transfer_config
        g = cfg.get_from_extra_config
        self.tail = int(g("tail", 256))
        self.tail_frac = float(g("tail_frac", 0.25))
        self.min_len = int(g("min_len", 512))
        self.src_name, self.maps_path, self.sink_path = g("src", None), g("maps", None), g("sink_template", None)
        self.src_sdpa = str(g("src_sdpa", "vllm_fa"))
        self.tgt_name = vllm_config.model_config.model
        self.max_len = vllm_config.model_config.max_model_len
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("kvx gemma connector: start vLLM with enable_prefix_caching=False")
        self.groups, self.layer_group = [], {}
        if kv_cache_config is not None:
            for gi, grp in enumerate(kv_cache_config.kv_cache_groups):
                ls = [int(_LAYER_RE.search(n).group(1)) for n in grp.layer_names]
                self.groups.append((int(grp.kv_cache_spec.block_size), ls))
                for l in ls:
                    self.layer_group[l] = gi
        self._need_load, self._m_req, self._ready, self._layouts = {}, {}, False, {}

    # ---------------- scheduler side
    def _n_mapped(self, n):
        if n < self.min_len:
            return 0
        t = max(1, min(self.tail, int(self.tail_frac * n)))
        return max(0, n - t)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        m = self._n_mapped(len(request.prompt_token_ids or []))
        self._m_req[request.request_id] = m
        return max(0, m - num_computed_tokens), False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens > 0:
            self._need_load[request.request_id] = request

    def build_connector_meta(self, scheduler_output):
        meta = GemmaMeta()
        for nr in scheduler_output.scheduled_new_reqs:
            if nr.req_id not in self._need_load:
                continue
            ids = list(nr.prompt_token_ids or [])
            m = self._m_req.pop(nr.req_id, None)
            m = self._n_mapped(len(ids)) if m is None else m
            meta.requests.append(_GReq(token_ids=ids[:m], block_ids=tuple(nr.block_ids)))
        self._need_load.clear()
        return meta

    def request_finished(self, request, block_ids):
        self._m_req.pop(request.request_id, None)
        return False, None

    def request_finished_all_groups(self, request, block_ids):
        self._m_req.pop(request.request_id, None)
        return False, None

    # ---------------- worker side
    def _tp(self):
        try:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
            return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        except Exception:
            return 0, 1

    def _init_worker(self, device):
        from transformers import AutoConfig
        from . import hybrid
        self.tp_rank, self.tp_size = getattr(self, "tp_override", None) or self._tp()
        if device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(float(getattr(self, "mem_fraction", 0.97)), device)
        self.Sdec = hybrid.load(self.src_name, attn="sdpa")
        self.Sdec.model.to(device)
        if self.src_sdpa == "vllm_fa":  # each layer's window + softcap through vLLM's varlen kernel (kvx/fa.py)
            from .fa import use_vllm_fa
            use_vllm_fa(self.Sdec.model)
        self.src_w = [dict(w) for w in hybrid.tap_widths(self.Sdec)]  # per source layer {"k": wk, "v": wv}
        tc = AutoConfig.from_pretrained(self.tgt_name)
        tc = getattr(tc, "text_config", tc)
        self.types = list(tc.layer_types)
        self.window = int(getattr(tc, "sliding_window", 0) or 0)
        rp = tc.rope_parameters
        mp = torch.load(self.maps_path, map_location="cpu")
        self.maps = {}
        for kind in ("k", "v"):
            for m in mp[kind]:
                t, H, D = int(m["layer"]), int(m["H"]), int(m["D"])
                Hl = H // self.tp_size if H >= self.tp_size else 1
                h0 = (self.tp_rank * Hl) if H >= self.tp_size else self.tp_rank * H // self.tp_size
                cols = torch.arange(h0 * D, (h0 + Hl) * D)
                self.maps.setdefault(t, {})[kind] = dict(
                    sel=[int(s) for s in m["sel"]], W=m["W"][:, cols].to(device, torch.bfloat16).contiguous(),
                    b=m["b"][cols].to(device, torch.float32), H=Hl, D=D, h0=h0)
        del mp
        self.kw = self._k_norm_weights(device)
        self.tabs = {}
        for ty in set(self.types):
            D = next(self.maps[t]["k"]["D"] for t in self.maps if self.types[t] == ty)
            self.tabs[ty] = gemma_rope_tables(rp[ty] if ty in rp else rp, D, self.max_len, device)
        self.sink = None
        if self.sink_path:
            st = torch.load(self.sink_path, map_location="cpu")
            self.sink = {int(t): {k: v.float().to(device) for k, v in d.items()} for t, d in st.items()}
        self._ready = True

    def _k_norm_weights(self, device):
        from safetensors import safe_open
        d = self.tgt_name if os.path.isdir(self.tgt_name) else __import__("huggingface_hub").snapshot_download(
            self.tgt_name, allow_patterns=["*.json", "*.safetensors"])
        ix = os.path.join(d, "model.safetensors.index.json")
        wm = json.load(open(ix))["weight_map"] if os.path.exists(ix) else None
        out = {}
        for t in self.maps:
            pat = re.compile(rf"(^|\.)layers\.{t}\.self_attn\.k_norm\.weight$")
            names = [n for n in (wm or {}) if pat.search(n)] if wm else None
            fn = os.path.join(d, wm[names[0]]) if wm else os.path.join(d, "model.safetensors")
            with safe_open(fn, "pt") as f:
                name = names[0] if wm else next(n for n in f.keys() if pat.search(n))
                out[t] = f.get_tensor(name).float().to(device)
        return out

    def _features(self, ids):
        from . import hybrid
        feats = hybrid.source_features(self.Sdec, ids)
        return feats  # per source layer [T, wk + wv] (k | v), None for KV-shared layers

    def _map_layer(self, t, feats, T):
        """Target layer t: (K [T, H, D] post-RoPE, V [T, H, D]) float32, this rank's heads."""
        out = []
        for kind in ("k", "v"):
            m = self.maps[t][kind]
            X = torch.cat([feats[s][:, :self.src_w[s]["k"]] if kind == "k" else feats[s][:, self.src_w[s]["k"]:]
                           for s in m["sel"]], 1)
            Y = (X.to(m["W"].dtype) @ m["W"]).float() + m["b"]
            out.append(Y.view(T, m["H"], m["D"]))
        K, V = out
        w = self.kw[t]
        K = restore_k(K, w)
        V = restore_v(V)
        if self.sink is not None and t in self.sink:
            A = min(T, self.sink[t]["k"].shape[0])
            mk, mv = self.maps[t]["k"], self.maps[t]["v"]
            sk = self.sink[t]["k"][:A].view(A, -1, mk["D"])[:, mk["h0"]:mk["h0"] + mk["H"]]
            sv = self.sink[t]["v"][:A].view(A, -1, mv["D"])[:, mv["h0"]:mv["h0"] + mv["H"]]
            K[:A], V[:A] = sk, sv
        cos, sin = self.tabs[self.types[t]]
        return _rope(K, cos[:T], sin[:T]), V

    def start_load_kv(self, forward_context, **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        if not isinstance(meta, GemmaMeta) or not meta.requests:
            return
        views = {}
        for n, l in forward_context.no_compile_layers.items():
            mm = _LAYER_RE.search(n)
            kvc = getattr(l, "kv_cache", None)
            if mm is None or kvc is None:
                continue
            views[int(mm.group(1))] = kvc[0] if isinstance(kvc, (list, tuple)) else kvc
            if not getattr(self, "_backend", ""):  # FA vs split4 is ambiguous by shape when H == 2 (31B global, TP2)
                be = getattr(l, "attn_backend", None)
                self._backend = str(be.get_name() if hasattr(be, "get_name") else type(getattr(l, "impl", None)).__name__)
        device = next(iter(views.values())).device
        if not self._ready:
            self._init_worker(device)
        self._write_requests(meta, views, device)

    def _write_requests(self, meta, views, device):
        from .vllm_connector import kv_layout, write_kv
        import time as _t
        for r in meta.requests:
            t0 = _t.perf_counter()
            T = len(r.token_ids)
            ids = torch.tensor([r.token_ids], device=device)
            feats = self._features(ids)
            t1 = _t.perf_counter()
            for t in sorted(self.maps):
                K, V = self._map_layer(t, feats, T)
                bs = self.groups[self.layer_group[t]][0]
                blocks = torch.tensor(r.block_ids[self.layer_group[t]], dtype=torch.long)
                pos = torch.arange(T)
                if self.types[t] == "sliding_attention" and self.window:
                    pos = pos[max(0, T - self.window):]  # only the window lives in the cache (null blocks before it)
                bi = blocks[pos // bs]
                ok = bi > 0  # skip vLLM's null block (id 0) if any slipped into the window
                pos, bi = pos[ok], bi[ok]
                if pos.numel() == 0:
                    continue
                off = pos % bs
                kv = views[t]
                key = (tuple(kv.shape), K.shape[1], K.shape[2])
                if key not in self._layouts:
                    self._layouts[key] = kv_layout(kv, K.shape[1], K.shape[2], getattr(self, "_backend", ""))
                pd = pos.to(device)
                write_kv(kv, bi.to(device), off.to(device), K[pd].to(torch.bfloat16), V[pd].to(torch.bfloat16), bs,
                         self._layouts[key])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            logger.info("kvx gemma: n=%d src_ms=%.1f map+write_ms=%.1f", T, 1e3 * (t1 - t0), 1e3 * (_t.perf_counter() - t1))

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        return

    def wait_for_save(self):
        return
