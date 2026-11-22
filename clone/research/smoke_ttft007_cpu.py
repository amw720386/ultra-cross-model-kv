# CPU smoke of research/ttft007.py with tiny random Llama models (Llama-3.1 tokenizer), short N.
import sys, torch, json
sys.path.insert(0, '.')
from transformers import LlamaConfig, LlamaForCausalLM
from research import ttft007
def tl(seed, hidden):
    torch.manual_seed(seed)
    return LlamaForCausalLM(LlamaConfig(vocab_size=128256, hidden_size=hidden, intermediate_size=128, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                            rope_theta=500000.0)).eval()
r = ttft007.run_pair("unsloth/Llama-3.1-8B-Instruct", "unsloth/Llama-3.1-8B-Instruct", N=128, models=(tl(0, 48), tl(1, 64)),
                     distill_steps=3, distill_lrs="1e-3", windows=2, prefix=48, warm=1, reps=3, out="/tmp/smoke_t007/r.json")
print(json.dumps(r["timing"])); print(json.dumps({a: v["kl32"] for a, v in r["kl32_greedy"].items()}))
