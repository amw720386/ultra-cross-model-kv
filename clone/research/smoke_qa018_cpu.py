# CPU smoke of research/qa018.py logic with tiny random models (Qwen tokenizer source -> TinyLlama tokenizer target),
# 2 items per task, LongBench zip fetched to /tmp.
import sys, torch, json
sys.path.insert(0, '.')
from transformers import Qwen3Config, Qwen3ForCausalLM, LlamaConfig, LlamaForCausalLM
from research import qa018
torch.manual_seed(0)
S = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=32768)).eval()
T = LlamaForCausalLM(LlamaConfig(vocab_size=32000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32768)).eval()
_lb = qa018.longbench
qa018.longbench = lambda task: [ex for ex in _lb(task, root="/tmp/lb") if len(ex["context"]) < 12000][:3]
r = qa018.run_pair("Qwen/Qwen3-0.6B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", n_calib=4, L=96, ks="1", models=(S, T),
                   n_per=2, max_ctx=6000, out="/tmp/smoke_qa018/r.json")
print(json.dumps({k: r[k] for k in ("native", "mapped", "r256", "s4r256")}))
