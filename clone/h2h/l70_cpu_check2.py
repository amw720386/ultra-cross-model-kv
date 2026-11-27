"""CPU check 2 (vLLM image): find vLLM 0.34's serve arg parser and parse every argv; compare source/target tokenizers."""
import json, subprocess, sys
res = {}
r = subprocess.run("grep -rln 'def make_arg_parser' $(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')/entrypoints | head",
                   shell=True, capture_output=True, text=True)
res["make_arg_parser_in"] = r.stdout.strip().splitlines()
argvs = json.loads(sys.argv[1])
mod = None
for f in res["make_arg_parser_in"]:
    m = f.split("site-packages/")[-1].replace("/", ".").removesuffix(".py")
    m = m[m.index("vllm."):] if "vllm." in m else m
    mod = m
    break
if mod:
    import importlib
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    p = importlib.import_module(mod).make_arg_parser(FlexibleArgumentParser())
    for name, argv in argvs.items():
        try:
            a = p.parse_args(argv[2:])
            res[name] = ["ok", a.model_tag if hasattr(a, "model_tag") else None, a.max_model_len, a.tensor_parallel_size,
                         a.kv_cache_dtype, a.gpu_memory_utilization, a.revision, getattr(a, "async_scheduling", None),
                         a.max_num_batched_tokens, bool(a.kv_transfer_config)]
        except SystemExit as e:
            res[name] = f"PARSE FAILED (SystemExit {e})"
from transformers import AutoTokenizer
T = "/hf/hub/models--RedHatAI--Llama-3.3-70B-Instruct-FP8-dynamic/snapshots/f50dbad2c84590ca17dc51e207c34321b65ff14b"
S = "/hf/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5"
tt, ts = AutoTokenizer.from_pretrained(T), AutoTokenizer.from_pretrained(S)
vt, vs = tt.get_vocab(), ts.get_vocab()
res["vocab_equal"] = vt == vs
res["vocab_sizes"] = [len(vt), len(vs)]
res["vocab_diff"] = sorted(set(vt.items()) ^ set(vs.items()))[:10]
txt = "The retrieval code is AMBER-7421. Llama 3.3 70B vs 3.1 8B: naïve café, 東京, emoji 🙂, code `x = f(y)`.\n" * 50
res["encode_equal"] = tt(txt)["input_ids"] == ts(txt)["input_ids"]
res["bos_eos"] = [[tt.bos_token, tt.eos_token], [ts.bos_token, ts.eos_token]]
msgs = [{"role": "user", "content": "hi"}]
ct = tt.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
cs = ts.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
res["chat_template_equal"] = ct == cs
res["chat_template_target"] = ct[:300]
res["config_target"] = {k: json.load(open(f"{T}/config.json")).get(k) for k in ("num_hidden_layers", "num_key_value_heads", "head_dim", "hidden_size", "num_attention_heads", "max_position_embeddings", "rope_scaling")}
print(json.dumps(res, indent=1, default=str))
