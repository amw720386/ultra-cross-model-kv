#!/bin/bash
# 2x A100-80GB chain: source + stats on cuda:0, target on cuda:1 (core.place_on).
export HF_HOME=/root/hf HF_HUB_ENABLE_HF_TRANSFER=1
cd /root/kvx
python run_local.py --src Qwen/Qwen3-1.7B --tgt Qwen/Qwen3-32B --ks 16,24 --modes rms --tails 64,256 --tail-k 24 --sinks 4 --n-calib 4000 --bench-n 200 --bench-tail 64 --save-maps 1 --run-id q17-q32-cal4000 > /root/run_a.log 2>&1
KVX_ONE_PASS=0 python run_local.py --src Qwen/Qwen3-14B --tgt Qwen/Qwen3-32B --ks 8,16 --modes rms --tails 64 --tail-k 16 --sinks 4 --n-calib 2000 --bench-n 200 --bench-tail 64 --run-id q14-q32-cal2000 > /root/run_b.log 2>&1
touch /root/chain.done
