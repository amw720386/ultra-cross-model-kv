# CPU smoke of KL distillation (tiny random Qwen3 pair): training lowers held-out KL; kl32 + mc128 wiring.
import sys, torch, json
sys.path.insert(0, '.')
from transformers import Qwen3Config, Qwen3ForCausalLM
from research import kl32, mc128
def tiny(layers, seed):
    torch.manual_seed(seed)
    c = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096)
    return Qwen3ForCausalLM(c).eval()
S, T = tiny(2, 0), tiny(3, 1)
r = kl32.run_pair("Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B", n_calib=8, L=128, ks="1,2", windows=3, prefix=96, steps=8,
                  recipes="", calib_src="fineweb", models=(S, T), dev="cpu", forced="text",
                  distill_steps=30, distill_lrs="1e-3,1e-2", distill_n=16)
print(json.dumps(r["distill"]["runs"]), r["distill"]["chosen"])
print({a: round(v["kl32"], 6) for a, v in r["kl32_text"].items()})
r = mc128.run_pair("Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B", n_calib=16, L=128, ks="1,2", n_heldout=4, per_task=2,
                   dtype="float32", models=(S, T), distill_steps=20, distill_lrs="1e-2", distill_n=16)
print(r.get("k_mc"), r["heldout"].get("map_kd"), {a: v["acc"] for a, v in r["mc128"].items()})
