#!/bin/bash
# GPU1 experiments while GPU0 serves the vLLM test: 1.7B->8B at 4000-seq calibration, feature variants.
export HF_HOME=/root/hf HF_HUB_ENABLE_HF_TRANSFER=1 CUDA_VISIBLE_DEVICES=1
cd /root/kvx
while [ ! -f /root/chain.done ]; do sleep 20; done
python run_local.py --src Qwen/Qwen3-1.7B --tgt Qwen/Qwen3-8B --ks 8,16,24 --modes rms --tails 64 --tail-k 24 --n-calib 4000 --run-id q17-q8-c4k-kv > /root/g1_kv.log 2>&1
python run_local.py --src Qwen/Qwen3-1.7B --tgt Qwen/Qwen3-8B --ks 8,16 --modes rms --tails 64 --tail-k 16 --n-calib 4000 --feat kvcat --run-id q17-q8-c4k-kvcat > /root/g1_kvcat.log 2>&1
python run_local.py --src Qwen/Qwen3-1.7B --tgt Qwen/Qwen3-8B --ks 4,8 --modes rms --tails 64 --tail-k 8 --n-calib 4000 --feat hidden --run-id q17-q8-c4k-hidden > /root/g1_hid.log 2>&1
touch /root/gpu1.done
