# CPU smoke of research/recall015.py with tiny random models (TinyLlama tokenizer -> Qwen3 tokenizer).
import sys, torch, json
sys.path.insert(0, '.')
from transformers import Qwen3Config, Qwen3ForCausalLM, LlamaConfig, LlamaForCausalLM
from research import recall015
torch.manual_seed(0)
S = LlamaForCausalLM(LlamaConfig(vocab_size=32000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)).eval()
T = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096)).eval()
recall015.QUESTIONS = {f: q[:1] for f, q in recall015.QUESTIONS.items()}  # 64 questions for speed
r = recall015.run_pair("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen3-0.6B", n_calib=4, L=96, ks="1", models=(S, T),
                       n_task=4, out="/tmp/smoke_r015/r.json")
print(json.dumps({k: v for k, v in r.items() if isinstance(v, dict) and "correct" in v}))
