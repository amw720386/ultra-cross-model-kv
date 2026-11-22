# CPU check: batched generation from an externally built cache == native batched generation (oracle KV, greedy).
import sys, torch
sys.path.insert(0, '.')
from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer
from research import mc128, genmap
torch.manual_seed(0)
c = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096)
T = Qwen3ForCausalLM(c).eval()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
P = mc128.Pair(T, T, tok, "cpu", Lmax=512)
prompts = [tok("Hello there, how are you doing today?")["input_ids"], tok("Short")["input_ids"] + [11, 12],
           tok("A somewhat longer prompt about the history of the Roman empire and its fall.")["input_ids"]]
kw = dict(max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
nat = genmap.generate(P, prompts, None, **kw)
one = [genmap.generate(P, [p], None, **kw)[0] for p in prompts]
for tail in (1, 2):
    orc = genmap.generate(P, prompts, P.native, tail=tail, **kw)
    print("tail", tail, "oracle==native", orc == nat, "batched==unbatched", nat == one)
    assert orc == nat and nat == one
print("OK")
