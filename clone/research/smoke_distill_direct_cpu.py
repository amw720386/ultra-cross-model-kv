# CPU smoke: kl32 distillation from a direct-injection init (tiny random Qwen3 pair, same KV shapes).
import sys, torch, json
sys.path.insert(0, '.')
from transformers import Qwen3Config, Qwen3ForCausalLM
from research import kl32
def tiny(layers, seed):
    torch.manual_seed(seed)
    c = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096)
    return Qwen3ForCausalLM(c).eval()
r = kl32.run_pair("Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B", n_calib=4, L=64, ks="1", windows=2, prefix=48, steps=8,
                  recipes="", calib_src="fineweb", models=(tiny(2, 0), tiny(3, 1)), dev="cpu", forced="text",
                  distill_steps=6, distill_lrs="1e-2", distill_n=8, distill_init="direct")
print(json.dumps(r["distill"]["runs"]), {a: round(v["kl32"], 6) for a, v in r["kl32_text"].items()})
