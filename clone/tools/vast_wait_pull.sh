#!/usr/bin/env bash
# Wait for run_local.py on a Vast box to exit, pull result+log, destroy the box, verify it is gone.
# usage: vast_wait_pull.sh <instance_id> <host> <port> <run_id>
set -u
ID=$1; H=$2; P=$3; RUN=$4
S="ssh -i $HOME/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $P root@$H"
OUT="$HOME/kvx-data/out/$RUN"; mkdir -p "$OUT"
while $S 'pgrep -f run_local.py >/dev/null' 2>/dev/null; do sleep 20; done
scp -i $HOME/.ssh/id_ed25519 -o StrictHostKeyChecking=no -P "$P" -q "root@$H:/vol/runs/$RUN/result.json" "$OUT/" 2>/dev/null
scp -i $HOME/.ssh/id_ed25519 -o StrictHostKeyChecking=no -P "$P" -q "root@$H:/root/run.log" "$OUT/run.log" 2>/dev/null
if [ "${KEEP:-0}" != 1 ]; then
  vastai destroy instance "$ID" >/dev/null 2>&1
  vastai show instances --raw 2>/dev/null | python3 -c "import json,sys;print('still listed' if any(i['id']==$ID for i in json.load(sys.stdin)) else 'destroyed')"
fi
ls -la "$OUT"; tail -c 600 "$OUT/run.log"
