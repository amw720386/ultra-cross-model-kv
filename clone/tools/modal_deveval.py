"""DevEval scoring in Modal Sandboxes (Python 3.9 image; Modal Functions reject < 3.11, Sandboxes do not).

prepare (network ON, runs only the repos' own reference code): extract Source_Code onto the volume, run the reference
check (fills each repo's .eggs with its test deps, as the official `setup.py pytest` does), write
/vol/data/deveval/ref.jsonl and the eligible list. score (network BLOCKED: model-generated code): copy the prepared
repos to local disk, score a generations file.

python tools/modal_deveval.py prepare
python tools/modal_deveval.py score /vol/runs/<run>/<arm>/gens.jsonl /vol/runs/<run>/<arm>/deveval_scored.jsonl
"""
import os
import sys
import time

import modal

APP = "kvx-qcpu-deveval"  # CPU sandboxes: outside the quality lanes' 2-GPU-app count
image = (modal.Image.from_registry("python:3.9-slim")
         .run_commands("pip install --no-cache-dir 'setuptools<70' pytest pytest-runner func_timeout "
                       "async-timeout==4.0.3 elementpath==4.1.5 fastapi==0.93.0 joblib==1.3.2 kinto-redis==2.1.0 "
                       "memory-profiler==0.61.0 numpy==1.26.2 pandas==2.1.2 pillow==10.1.0 psutil==5.9.7 "
                       "pyarrow==14.0.0 pydantic==1.10.13 pytz==2023.3.post1 redis==5.0.1 scikit-learn==1.3.2 "
                       "scipy==1.11.3 starlette==0.25.0 threadpoolctl==3.2.0 tzdata==2023.3 xmlschema==2.5.1 "
                       "dill==0.3.7 jinja2==3.1.3"))
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"))
D = "/vol/data/deveval"


def _run(cmd, network, timeout, cpu=8):
    app = modal.App.lookup(APP, create_if_missing=True)
    sb = modal.Sandbox.create("bash", "-c", cmd, image=image, app=app, volumes={"/vol": vol}, timeout=timeout,
                              cpu=cpu, memory=32768, block_network=not network)
    t0 = time.time()
    sb.wait()
    out, err = sb.stdout.read(), sb.stderr.read()
    print(out[-4000:], err[-3000:], "rc", sb.returncode, "wall", round(time.time() - t0), flush=True)
    return sb.returncode


def prepare():
    cmd = (f"set -e; cd {D}; [ -d Source_Code ] || tar -xzf Source_Code.tar.gz; "
           f"python deveval_score.py --data data.jsonl --src Source_Code --reference --out ref.jsonl --workers 8; "
           "python -c \"import json; "
           "r=[json.loads(l) for l in open('ref.jsonl')]; ok={x['id'] for x in r if x['result']=='Pass'}; "
           "order=[json.loads(l)['namespace'] for l in open('data.jsonl')]; "
           "open('deveval_eligible.txt','w').write(''.join(n+chr(10) for n in order if n in ok)); print(len(ok))\"; sync")
    return _run(cmd, network=True, timeout=3 * 3600)


def score(gens, out):
    cmd = (f"set -e; mkdir -p /tmp/dv && cp -r {D}/Source_Code /tmp/dv/ && cd /tmp/dv && "
           f"python {D}/deveval_score.py --data {D}/data.jsonl --src /tmp/dv/Source_Code --completions {gens} "
           f"--only {D}/deveval_eligible.txt --out /tmp/dv/scored.jsonl --workers 8 && cp /tmp/dv/scored.jsonl {out}; sync")
    return _run(cmd, network=False, timeout=2 * 3600)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        sys.exit(prepare())
    sys.exit(score(sys.argv[2], sys.argv[3]))
