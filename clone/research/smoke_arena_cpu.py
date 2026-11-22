# CPU smoke of research/arena.py with tiny random Qwen3 models (fetches the 100 OneTriangle prompts).
import sys, torch, json
sys.path.insert(0, '.')
from transformers import Qwen3Config, Qwen3ForCausalLM
from research import arena
def tiny(layers, seed):
    torch.manual_seed(seed)
    c = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=8192)
    return Qwen3ForCausalLM(c).eval()
r = arena.run_pair("Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B", n_calib=8, L=128, ks="1,2", models=(tiny(2, 0), tiny(3, 1)),
                   out="/tmp/smoke_arena/r.json", bs=50, max_new=12, recipes="s4t64")
print(json.dumps({k: r[k] for k in ("k_selected", "prompt_tokens", "native8b", "mapped", "recipe_s4t64")}))
