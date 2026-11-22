# CPU smoke of research/xfit.py: (1) same model + same tokenizer -> identity alignment -> mapped == native;
# (2) tiny Llama-tokenizer source -> tiny Qwen target runs end to end.
import sys, torch, json
sys.path.insert(0, '.')
from transformers import Qwen3Config, Qwen3ForCausalLM, LlamaConfig, LlamaForCausalLM
from research import xfit
def tq(layers, seed):
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096)).eval()
def tl(seed):
    torch.manual_seed(seed)
    return LlamaForCausalLM(LlamaConfig(vocab_size=128256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)).eval()
m = tq(2, 0)
r = xfit.run_pair("Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B", n_calib=6, L=96, ks="1,2", models=(m, m), n_heldout=3, out="/tmp/smoke_xfit/a.json")
print("same:", json.dumps(r["heldout"]))
assert r["heldout"]["map_k1"]["kl"] < 1e-5
r = xfit.run_pair("unsloth/Llama-3.1-8B-Instruct", "Qwen/Qwen3-0.6B", n_calib=6, L=96, ks="1,2", models=(tl(1), tq(3, 2)), n_heldout=3, out="/tmp/smoke_xfit/b.json")
print("cross:", json.dumps(r["heldout"]), r["bos_s"], r["bos_t"])
import torch as t; d = t.load("/tmp/smoke_xfit/b_maps_k%d.pt" % r["k_selected"]); print(sorted(d), d["src"], len(d["k"]))
