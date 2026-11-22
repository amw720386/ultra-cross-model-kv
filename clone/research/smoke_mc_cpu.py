# CPU smoke: tiny random Qwen3 models; `python research/smoke_mc_cpu.py same` -> every arm must equal native (choice KL ~0).
import sys, torch, json
sys.path.insert(0, '.')
from transformers import Qwen3Config, Qwen3ForCausalLM
from research import mc128
def tiny(layers, seed):
    torch.manual_seed(seed)
    c = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096)
    return Qwen3ForCausalLM(c).eval()
S, T = (tiny(2, 0), tiny(3, 1)) if len(sys.argv) < 2 else (tiny(2, 0), tiny(2, 0))
r = mc128.run_pair("Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B", n_calib=8, L=128, ks="1,2", n_heldout=4, per_task=3,
                   hs_n=4, piqa_n=4, calib_src="mix", recipes="s4t16,s0t8", out="/tmp/smoke_mc/r.json", dtype="float32", models=(S, T))
print(json.dumps({k: r[k] for k in ("k_selected", "heldout", "mc128", "piqa")}, indent=0)[:3000])
