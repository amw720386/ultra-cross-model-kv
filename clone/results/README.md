# clone/results/

Raw `result.json` output of each `kvx` run behind the tables in [../../docs/onetriangle-clone.md](../../docs/onetriangle-clone.md).
One file per run id (`<src>-<tgt>-<variant>.json`). Each records `src`, `tgt`, calibration settings, and the KL / accuracy
evals per mapper variant.

- `vllm-*/`: vLLM 0.34 serving runs, native vs `CrossModelConnector` (TTFT rows and first-token logprobs).
  `vllm-72b-fa2` is Qwen2.5-3B → 72B on one B200, `vllm-72b-128k-tp2` the 128K tensor-parallel run,
  `vllm-32b-128k` Qwen3-1.7B → 32B at 128K.
- Everything else: HF-based fit and eval runs (`q17-q32-*` Qwen3-1.7B → 32B, `q17-q8-*` → 8B, `q14-q32-*`, `q25-*`
  Qwen2.5, `g2/g3-*` Gemma, `l*` Llama, `mi8-ms24`, `olmo2-*`, `phi3-*`, `dsd-*`).
- The fitted maps (`maps_k*.pt`) are not included; they were not preserved. Refit with `modal_fit_eval.py`.
