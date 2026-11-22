#!/bin/bash
# Job chain for a rented H200 box (no idle gaps). Results land in /vol/runs/<run_id>/.
export HF_HOME=/root/hf HF_HUB_ENABLE_HF_TRANSFER=1
cd /root/kvx
python run_local.py --src Qwen/Qwen3-1.7B --tgt Qwen/Qwen3-32B --ks 16,24 --modes rms --tails 64,256 --tail-k 24 --sinks 4 --n-calib 4000 --bench-n 200 --bench-tail 64 --save-maps 1 --run-id q17-q32-cal4000 > /root/run_a.log 2>&1
python -m kvx.ttft --src Qwen/Qwen3-1.7B --tgt Qwen/Qwen3-32B --maps /vol/runs/q17-q32-cal4000/maps_k24.pt --lens 4096,8192,16384,32768 --tail 64 --dense 1 --out /vol/runs/q17-q32-cal4000/ttft_h200nvl_dense.json > /root/ttft_d.log 2>&1
python -m kvx.ttft --src Qwen/Qwen3-1.7B --tgt Qwen/Qwen3-32B --maps /vol/runs/q17-q32-cal4000/maps_k24.pt --lens 4096,8192 --tail 64 --dense 0 --out /vol/runs/q17-q32-cal4000/ttft_h200nvl_perlayer.json > /root/ttft_p.log 2>&1
python run_local.py --src Qwen/Qwen3-1.7B --tgt Qwen/Qwen3-8B --ks 24,32,40 --modes rms --tails 64 --tail-k 32 --n-calib 16000 --run-id q17-q8-calib16000 > /root/run_b.log 2>&1
touch /root/chain.done
