# Reproduce

Everything runs on [Modal](https://modal.com). From `modal_jobs/` with `PYTHONPATH=../src`.

## Secrets and switches

- Official τ2 simulator and Terminal-Bench need Modal secrets `xkv-openai` (`OPENAI_API_KEY`) and `xkv-harbor-modal`
  (a Modal token for the task sandboxes).
- Thinking-on rows: `"env":{"XKV_THINKING":"1"}`. QMSum judge: `"env":{"XKV_QMSUM_JUDGE":"1"}`.
- Code benchmarks: `"env":{"XKV_CODE_PROMPT":"free"}`. BigCodeBench runs in the official image:
  `modal deploy exec_app.py`, then `"exec_modal":true`.
- Graphed tail off: `XKV_NO_GRAPH_TAIL=1`.

## Speed

```bash
# TTFT, 4B → 27B, fast + recall
modal run lab.py --task ttft_repair --gpu H100 --args '{"repairs":[0,1024,2048]}'
# TTFT, 2B → 27B
modal run lab.py --task ttft_repair --gpu H100 --args '{"src":"2b","mapper_name":"2b-27b-fid+2b-27b-fid-attn","repairs":[0,1024]}'
modal run lab.py --task ttft_repair --gpu H100 --args '{"src":"2b","mapper_name":"2b-27b-fid+2b-27b-fid-attn","repairs":[0,2048,"2048m","3072m"],"msg_every":400}'
# decode
modal run lab.py --task decode_bench --gpu H200 --args '{"lengths":[2048,32768],"batches":[1,2,3,4,8,12,16,32]}'
# serving
modal run lab.py --task serve --gpu H200 --args '{}'
```

## Quality

```bash
# 4B → 27B suite
modal run lab.py --task gen_evals --gpu H200 --args '{"mapper_name":"4b-27b-fid+4b-27b-fid-attn","full_until":32768,"benches":["contract_nli","longhealth"],"strategies":["target","transfer@auto"]}'
# 2B → 27B RULER / MRCR, fast + recall
modal run lab.py --task gen_evals --gpu H200 --args '{"src":"2b","mapper_name":"2b-27b-fid+2b-27b-fid-attn","full_until":32768,"max_ctx":32768,"benches":["ruler","mrcr"],"strategies":["transfer@auto","transfer@auto+rep1024"]}'
# MRCR held-out
modal run lab.py --task gen_evals --gpu H200 --args '{"src":"2b","mapper_name":"2b-27b-fid+2b-27b-fid-attn","full_until":32768,"max_ctx":32768,"benches":["mrcr"],"strategies":["transfer@auto+rep3072m"],"limits":{"mrcr":120},"bench_kw":{"mrcr":{"min_ctx":4096,"batch":4,"offset":60}}}'
# held-out fidelity gate
modal run lab.py --task quality --gpu H100 --args '{"mapper_name":"4b-27b-fid+4b-27b-fid-attn","n_seqs":32,"prefix":4096,"tails":[1,128],"mc_limit":300,"mc_tails":[1]}'
```

## Agents

```bash
# log native τ2 requests (replay data)
modal run agent_lab.py --task tau2 --gpu H200 --args '{"strategies":["target"],"domains":["airline"],"full_until":32768,"log_requests":true,"run_name":"replay"}'
# agent setting
modal run agent_lab.py --task tau2 --gpu H200 --args '{"mapper_name":"4b-27b-fid+4b-27b-fid-attn","full_until":32768,"strategies":["transfer@auto+rep4096m"],"domains":["airline","retail","telecom"],"turn_tail_max":8192,"turn_tail_msgs":10,"native_tools":true,"local_user_and_judges":false}'
# policy-deleted control
modal run agent_lab.py --task tau2 --gpu H200 --args '{"full_until":32768,"strategies":["target-drop"],"domains":["airline","retail","telecom"],"turn_tail_max":8192,"turn_tail_msgs":10,"local_user_and_judges":false}'
# train the agent map
modal run lab.py --task distill --gpu H200 --args '{"mapper_name":"4b-27b-fid+4b-27b-fid-attn","out_name":"4b-27b-fid-agent","steps":600,"prefix_choices":[2048,4096,8192,12288],"prefix":8192,"cont":128,"tail":64,"eval_tail":128,"chat_frac":0.15,"tool_frac":0.15,"retrieval_frac":0.3,"varied_frac":0.45,"copy_frac":0.35,"agent_logs":["tau2/replay/target/airline_requests.jsonl"],"agent_frac":0.4,"agent_tail_max":1024,"agent_tail_msgs":2,"eval_agent":24,"lr":1e-5,"warmup":20,"eval_every":200,"n_eval":4,"niah_len":16384,"niah_n":10}'
# lighter agent setting
modal run agent_lab.py --task tau2 --gpu H200 --args '{"mapper_name":"4b-27b-fid-agent+4b-27b-fid-agent-attn","full_until":32768,"strategies":["transfer@auto"],"domains":["airline","retail","telecom"],"turn_tail_max":4096,"turn_tail_msgs":3,"native_tools":true,"local_user_and_judges":false}'
# replay with TTFT per variant
modal run agent_lab.py --task tau2_replay --gpu H200 --args '{"domains":["telecom","retail"],"max_prompts":160,"max_new":128,"ttft_n":16,"variants":[{"name":"native","strategy":"target"},{"name":"agent","strategy":"transfer@auto+rep4096m","native_tools":true,"turn_tail_max":8192,"turn_tail_msgs":10},{"name":"lighter_nograph","strategy":"transfer@auto","native_tools":true,"turn_tail_max":4096,"turn_tail_msgs":3,"mapper":"4b-27b-fid-agent+4b-27b-fid-agent-attn","env":{"XKV_NO_GRAPH_TAIL":"1"}},{"name":"deleted","strategy":"target","drop_mapped":true,"turn_tail_max":8192,"turn_tail_msgs":10}]}'
# Terminal-Bench 2.0
modal run agent_lab.py --task terminal_bench --gpu H200 --args '{"mapper_name":"4b-27b-fid+4b-27b-fid-attn","full_until":32768,"strategies":["target","transfer@auto+rep4096m"],"task_names":["gpt2-codegolf","break-filter-js-from-html","llm-inference-batching-scheduler","write-compressor","merge-diff-arc-agi-task","pytorch-model-cli","reshard-c4-data","largest-eigenval","password-recovery","log-summary-date-ranges"],"concurrency":10,"turn_tail_max":8192,"turn_tail_msgs":10,"agent_timeout_multiplier":2}'
```
