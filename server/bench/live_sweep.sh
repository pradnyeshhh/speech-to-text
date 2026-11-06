#!/bin/bash
# usage: live_sweep.sh LABEL MODEL [server options...]
here=$(cd "$(dirname "$0")" && pwd); s=${TMPDIR:-/tmp}/voice-bench; mkdir -p "$s"; label=$1; model=$2; shift 2
cd /root/voice_to_text; rm -rf $s/sweepdata; log=$s/sweep-server.log
HF_HUB_OFFLINE=1 nice -n 10 .venv/bin/python -u server/server.py --model $model --socket $s/sweep.sock --data $s/sweepdata "$@" > $log 2>&1 &
pid=$!
for i in $(seq 1 240); do grep -qE 'Listening on|Traceback' $log && break; sleep 1; done
for clip in long medium; do echo "$label $(.venv/bin/python -I $here/live_measure.py $s/sweep.sock $clip)"; done
kill $pid; wait $pid 2>/dev/null
