import sys, torch, json
sys.path.insert(0, '.')
from transformers import Qwen3Config, Qwen3ForCausalLM
from research import kl32
def tiny(layers, seed):
    torch.manual_seed(seed)
    c = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096)
    return Qwen3ForCausalLM(c).eval()
S, T = (tiny(2, 0), tiny(3, 1)) if len(sys.argv) < 2 else (tiny(2, 0), tiny(2, 0))
r = kl32.run_pair("Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B", n_calib=8, L=128, ks="1,2", windows=3, prefix=96, steps=8,
                  recipes="s4t16", calib_src="mix", out="/tmp/smoke_kl32/r.json", models=(S, T), dev="cpu")
print(json.dumps({f: {a: (round(v["kl32"], 7), v["agree"]) for a, v in r[f].items()} for f in ("kl32_greedy", "kl32_text")}))
