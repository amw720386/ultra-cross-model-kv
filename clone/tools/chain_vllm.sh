#!/bin/bash
# vLLM native vs CrossModelConnector on one GPU of a rented box (separate venv: vLLM pins its own torch).
set -x
export HF_HOME=/root/hf HF_HUB_ENABLE_HF_TRANSFER=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN CUDA_VISIBLE_DEVICES=0,1 KVX_VLLM_UTIL=0.85 KVX_SRC_DEVICE=cuda:1 LENS=4096,8192,16384
python -m venv /root/vv && /root/vv/bin/pip install -q vllm==0.34.0 hf_transfer > /root/vv_install.log 2>&1
cd /root/kvx
MAPS=${MAPS:-/vol/runs/q17-q32-cal4000/maps_k24.pt}
for MODE in ${MODES:-native mapped}; do
/root/vv/bin/python -c "
import json, os, modal_vllm as m
r = m.serve_impl('$MODE', 'Qwen/Qwen3-1.7B', 'Qwen/Qwen3-32B', '$MAPS', os.environ.get('LENS', '4096,8192,16384'), 64)
print(json.dumps({k: r[k] for k in ('mode', 'gpu', 'rows')}), flush=True)
" >> /root/vllm_run.log 2>&1
done
touch /root/vllm.done
