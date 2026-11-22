#!/bin/bash
# Short TTFT chain (< 20 min after boot): quick k16 fit for maps, then HF TTFT dense + per-layer.
export HF_HOME=/root/hf HF_HUB_ENABLE_HF_TRANSFER=1
cd /root/kvx
python run_local.py --src Qwen/Qwen3-1.7B --tgt Qwen/Qwen3-32B --ks 16 --modes rms --n-calib 500 --n-eval 8 --save-maps 1 --run-id q17-q32-ttftmaps > /root/run_m.log 2>&1
python -m kvx.ttft --src Qwen/Qwen3-1.7B --tgt Qwen/Qwen3-32B --maps /vol/runs/q17-q32-ttftmaps/maps_k16.pt --lens 4096,8192,16384,32768 --tail 64 --dense 1 --profile 1 --out /vol/runs/q17-q32-ttftmaps/ttft_h200_dense.json > /root/ttft_d.log 2>&1
python -m kvx.ttft --src Qwen/Qwen3-1.7B --tgt Qwen/Qwen3-32B --maps /vol/runs/q17-q32-ttftmaps/maps_k16.pt --lens 4096,8192 --tail 64 --dense 0 --out /vol/runs/q17-q32-ttftmaps/ttft_h200_perlayer.json > /root/ttft_p.log 2>&1
touch /root/chain.done
