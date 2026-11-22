"""Cross-family ridge maps with the research/xalign contract (notes 015 / 017 / 018).

Fit pairs (X, Y) = (source pre-RoPE K/V at the aligned source index, target pre-RoPE K/V at target position j) on
text calibration (FineWeb-Edu + PG19 train), both models reading the same character prefix with their own tokenizers.
Saves maps in the kvx format (k / v lists, src, tgt, k_layers, feat) plus the alignment rule name, for the connector.
"""
import json, math, os, time

import torch

from kvx import core
from research import mc128
from research.xalign import encode_pair


def texts_fineweb(n, skip=0, stop=None, min_chars=3000):
    from datasets import load_dataset
    out = []
    for i, ex in enumerate(load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)):
        if i < skip:
            continue
        if stop is not None and i >= stop:
            break
        if len(ex["text"]) >= min_chars:
            out.append(ex["text"])
            if len(out) >= n:
                break
    return out


def texts_pg19(n, chunk=6000, per_book=16):
    from datasets import load_dataset
    out = []
    for ex in load_dataset("emozilla/pg19", split="train", streaming=True):
        t = ex["text"]
        for c in range(min(per_book, len(t) // chunk)):
            out.append(t[c * chunk:(c + 1) * chunk])
            if len(out) >= n:
                return out
    return out


class XPair(mc128.Pair):
    def __init__(self, S, T, tok_s, tok_t, dev, dev_t=None, Lmax=4096, bos_s=None, bos_t=None):
        self.dev_t = dev_t or dev
        if self.dev_t != dev:
            core.place_on(T, self.dev_t, dev)
        self.S, self.T, self.tok, self.tok_s, self.dev = S, T, tok_t, tok_s, dev
        self.bos_s, self.bos_t = bos_s, bos_t
        self.cs, self.ss = core.rope_cos_sin(S, Lmax, dev)
        ct, st = core.rope_cos_sin(T, Lmax, self.dev_t)
        self.ct, self.st = ct.to(dev), st.to(dev)
        hd = lambda c: getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        self.Ht, self.Dt, self.Lt = T.config.num_key_value_heads, hd(T.config), T.config.num_hidden_layers
        self.Hs, self.Ds, self.Ls = S.config.num_key_value_heads, hd(S.config), S.config.num_hidden_layers
        try:
            self.kw = [l.self_attn.k_norm.weight.detach().float().to(dev) for l in T.model.layers]
        except AttributeError:
            self.kw = None
        self.maps = {}

    def enc(self, text, n_tgt):
        return encode_pair(self.tok_s, self.tok, text, n_tgt, self.bos_s, self.bos_t)

    @torch.no_grad()
    def kv_pair(self, text, n_tgt):
        s_ids, t_ids, idx = self.enc(text, n_tgt)
        Ks, Vs, _ = core.capture_kv(self.S, torch.tensor([s_ids], device=self.dev), self.cs, self.ss)
        idx = idx.to(self.dev)
        return Ks[:, :, :, idx], Vs[:, :, :, idx], t_ids

    def fit_texts(self, texts, ks, L=512, lam_rel=1e-3, stride=4):
        r2 = {}
        for kind in ("k", "v"):
            st = core.GramStats(self.Ls, self.Hs * self.Ds, self.Lt, self.Ht * self.Dt, self.dev)
            for text in texts:
                Ka, Va, t_ids = self.kv_pair(text, L)
                with torch.no_grad():
                    Kt, Vt, _ = core.capture_kv(self.T, torch.tensor([t_ids], device=self.dev), self.ct, self.st)
                X = (Ka if kind == "k" else Va)[:, :, :, ::stride]
                Y = (Kt if kind == "k" else Vt)[:, :, :, ::stride]
                st.add(core.flat_tokens(X).reshape(-1, self.Ls * self.Hs * self.Ds).float(),
                       core.flat_tokens(Y).reshape(-1, self.Lt * self.Ht * self.Dt).float())
                del Ka, Va, Kt, Vt, X, Y
            st.finalize()
            R2 = core.layer_r2(st, 0.0, self.Ht, lam_rel)
            r2[kind] = {"best_single": R2.max(-1).values.mean().item()}
            for k in ks:
                ms = core.fit_maps(st, R2, k, 0.0, self.Ht, lam_rel)
                for m in ms:
                    m["W"] = m["W"].to(torch.bfloat16)
                self.maps[(kind, k)] = ms
                r2[kind][f"k{k}"] = torch.stack([m["r2"] for m in ms]).mean().item()
            del st, R2
            torch.cuda.empty_cache()
        return r2

    @torch.no_grad()
    def heldout_x(self, texts, P=448, C=64, ks=(), restore="rms"):
        """Target sees mapped prefix (first P target tokens, source read the same characters) + native continuation."""
        acc = {}
        for text in texts:
            Ka, Va, t_pre = self.kv_pair(text, P)
            _, t_full, _ = self.enc(text, P + C + (1 if self.bos_t is not None else 0))
            if len(t_full) < len(t_pre) + C:
                continue
            full = torch.tensor([t_full[:len(t_pre) + C]], device=self.dev)
            cont = full[:, len(t_pre):]
            lg_ref = self.T(input_ids=full).logits[:, len(t_pre) - 1:-1].float()
            lp_ref = torch.log_softmax(lg_ref, -1)
            top = lg_ref.argmax(-1)
            tin = full[:, len(t_pre) - 1:-1]  # last prefix token + cont[:-1], natively

            def add(name, lg):
                lp = torch.log_softmax(lg.float(), -1)
                a = acc.setdefault(name, dict(kl=0.0, agree=0.0, n=0))
                a["kl"] += (lp_ref.exp() * (lp_ref - lp)).sum(-1).mean().item()
                a["agree"] += (lg.argmax(-1) == top).float().mean().item()
                a["n"] += 1
            add("nocontext", self.T(input_ids=tin).logits)
            for k in ks:
                Km, Vm = self.mapped(Ka[:, :, :, :-1], Va[:, :, :, :-1], k, restore)
                add(f"map_k{k}", self.T(input_ids=tin, past_key_values=core.build_cache(Km, Vm, self.ct, self.st)).logits)
        return {n: dict(kl=a["kl"] / a["n"], agree=a["agree"] / a["n"], n=a["n"]) for n, a in acc.items()}

    def save_maps(self, path, k):
        cpu = lambda ms: [{a: (b.float().cpu() if torch.is_tensor(b) else b) for a, b in m.items()} for m in ms]
        torch.save({"k": cpu(self.maps[("k", k)]), "v": cpu(self.maps[("v", k)]), "src": self.src, "tgt": self.tgt,
                    "k_layers": k, "feat": "kv", "align": "xalign.align: target token j <- source token containing "
                    "the last character of j", "bos_s": self.bos_s, "bos_t": self.bos_t}, path)


def run_pair(src, tgt, n_calib=1000, L=512, ks="8,16,24", restore="auto", out="", models=None, n_heldout=32, **_):
    from transformers import AutoTokenizer
    from kvx.models import model_path
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev_t = "cuda:1" if torch.cuda.device_count() > 1 else dev
    t0 = time.time()
    tok_s, tok_t = AutoTokenizer.from_pretrained(model_path(src)), AutoTokenizer.from_pretrained(model_path(tgt))
    bos = lambda tk, name: tk.bos_token_id if (tk.bos_token_id is not None and "qwen" not in name.lower()) else None
    S, T = models if models else (mc128.load(src, dev, torch.bfloat16), mc128.load(tgt, dev_t, torch.bfloat16))
    P = XPair(S, T, tok_s, tok_t, dev, dev_t, bos_s=bos(tok_s, src), bos_t=bos(tok_t, tgt))
    P.src, P.tgt = src, tgt
    restore = restore if restore != "auto" else ("rms" if P.kw is not None else "none")
    klist = [min(int(k), P.Ls) for k in str(ks).split(",")]
    texts = texts_fineweb(n_calib // 2, stop=mc128.EVAL_SKIP) + texts_pg19(n_calib - n_calib // 2)
    res = dict(src=src, tgt=tgt, n_calib=len(texts), L=L, ks=klist, restore=restore, align="last-char-containment",
               bos_s=P.bos_s, bos_t=P.bos_t, t_load=time.time() - t0)
    res["r2"] = P.fit_texts(texts, klist, L)
    res["t_fit"] = time.time() - t0
    print(f"[{time.time()-t0:.0f}s] fit {res['r2']}", flush=True)
    ev = texts_fineweb(n_heldout, skip=mc128.EVAL_SKIP)
    res["heldout"] = P.heldout_x(ev, L - 64, 64, klist, restore)
    print(f"[{time.time()-t0:.0f}s] heldout {json.dumps(res['heldout'])}", flush=True)
    kbest = min(klist, key=lambda k: res["heldout"][f"map_k{k}"]["kl"])
    res["k_selected"] = kbest
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        P.save_maps(out.replace(".json", f"_maps_k{kbest}.pt"), kbest)
        json.dump(res, open(out, "w"), indent=1)
    return res
