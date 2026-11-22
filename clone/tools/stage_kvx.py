"""Weights stager: put a model into the shared Modal weights volume on a CPU container (no GPU clock).

The shared volume is KVX_SHARED_WEIGHTS_VOLUME (HF hub-cache layout under /hub). A model is staged when
/hub/models--org--name/.staged exists; that marker is written last, after every snapshot file resolves at its source
size (copy) or the Hub download finished (hub).

    modal run stage.py --repos org/name[,org/name...]              # copy from a seat volume that holds a complete
                                                                   # copy (no egress), else download from the Hub
    modal run stage.py --repos org/name --src some-volume  # copy from this volume only
    modal run stage.py --repos org/name --hub                      # Hub download only (gated repos need HF_TOKEN)

One container per repo, in parallel, <= 55 min each (the box reaper's cap); rerun to resume: files already present at
the right size are skipped. Sources are mounted read-only; staging never deletes anything.

    modal run stage.py::prune --vol <your seat volume> --repos org/name,...   # plan: what dropping your copies frees
    modal run stage.py::prune --vol <your seat volume> --repos ... --yes      # delete them

prune deletes a repo folder and the hub-level blobs only it references, and only when the shared volume holds a staged
copy of the same revision with every file the same size. Run it on your own volume, when none of your apps mounts it.
"""
import json, os, shutil, subprocess, time
import modal

SHARED = os.environ.get("KVX_SHARED_WEIGHTS_VOLUME", "")   # the shared read-only weights volume, if any
# Volumes that may hold a complete copy, in preference order: KVX_STAGE_SOURCES="vol1,vol2,..."
SOURCES = [v for v in os.environ.get("KVX_STAGE_SOURCES", os.environ.get("KVX_VOLUME", "kvx")).split(",") if v]
# kvx keeps its HF cache under hf/ (HF_HOME=/vol/hf), so its hub dir is <volume>/hf/hub
VOL_PREFIX = {os.environ.get("KVX_VOLUME", "kvx"): "hf"}
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".gguf")

img = (modal.Image.debian_slim(python_version="3.12")
       .pip_install("huggingface_hub[hf_xet]>=0.34")
       .env({"HF_HOME": "/dst", "HF_XET_HIGH_PERFORMANCE": "1"}))
app = modal.App(os.environ.get("KVX_APP", "kvx-stage-prune"))
dst_vol = modal.Volume.from_name(SHARED)
mounts = {"/dst": dst_vol, **{f"/src/{n}": modal.Volume.from_name(n).read_only() for n in SOURCES}}


def repo_dir(root, repo):
    return f"{root}/hub/models--{repo.replace('/', '--')}"


def snapshot_files(rd):
    """{path relative to the repo dir: resolved size, or None when the link does not resolve} for every snapshot."""
    out = {}
    for dp, _, fs in os.walk(f"{rd}/snapshots"):
        for f in fs:
            p = os.path.join(dp, f)
            out[os.path.relpath(p, rd)] = os.path.getsize(p) if os.path.exists(p) else None
    return out


def complete(rd):
    """Complete = a snapshot holds weight files and every snapshot file resolves to a non-empty file."""
    files = snapshot_files(rd) if os.path.isdir(f"{rd}/snapshots") else {}
    if not (any(k.endswith(WEIGHT_EXT) for k in files) and all(files.values())):
        return False
    for k in files:  # a sharded checkpoint must hold every shard its index names
        if k.endswith(".safetensors.index.json"):
            snap = os.path.dirname(k)
            with open(os.path.join(rd, k)) as fh:
                shards = set(json.load(fh).get("weight_map", {}).values())
            if any(os.path.join(snap, f) not in files for f in shards):
                return False
    return True


def copy_file(src, dst):
    if os.path.isfile(dst) and not os.path.islink(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + ".stage-tmp")
    os.replace(dst + ".stage-tmp", dst)
    return os.path.getsize(dst)


def copy_repo(src_root, repo):
    """Mirror hub/models--X into /dst with symlinks kept verbatim, plus each file outside the repo dir that a link
    resolves to (the hub-level blob store hub/blobs/<2>/<hash>, shared by repos). Returns bytes written."""
    srd, drd, written = repo_dir(src_root, repo), repo_dir("/dst", repo), 0
    for dp, dns, fs in os.walk(srd):
        rel = os.path.relpath(dp, srd)
        os.makedirs(os.path.join(drd, rel), exist_ok=True)
        for f in fs + [d for d in dns if os.path.islink(os.path.join(dp, d))]:
            if f.endswith((".incomplete", ".stage-tmp")) or f == ".staged":
                continue
            s, d = os.path.join(dp, f), os.path.join(drd, rel, f)
            if not os.path.islink(s):
                written += copy_file(s, d)
                continue
            # Resolve one hop by path, not realpath: the mount point itself is a link to /__modal/volumes/<id>.
            target = os.readlink(s)
            hop = os.path.normpath(os.path.join(os.path.dirname(s), target))
            if not hop.startswith(srd + "/") and os.path.isfile(hop):
                written += copy_file(hop, f"/dst/hub/{os.path.relpath(hop, f'{src_root}/hub')}")
            if os.path.islink(d) and os.readlink(d) == target:
                continue
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(target, d)
    return written


# max_containers: the workspace shares ~10 containers across all seats, and a v1 volume wants <= 5 writers.
@app.function(image=img, volumes=mounts, timeout=55 * 60, cpu=4, memory=8 * 1024, max_containers=3)
def stage(repo: str, src: str, hub_only: bool):
    t0 = time.time()
    dst_vol.reload()
    drd = repo_dir("/dst", repo)
    if os.path.exists(f"{drd}/.staged") and complete(drd):
        return f"{repo}: already staged"
    how = "hub"
    if not hub_only:
        for name in ([src] if src else SOURCES):
            if complete(repo_dir(f"/src/{name}", repo)):
                how = f"copy:{name}"
                break
        if how == "hub" and src:
            return f"{repo}: {src} holds no complete copy; nothing done"
    if how == "hub":
        p = subprocess.run(["hf", "download", repo, "--max-workers", "16"], capture_output=True, text=True)
        if p.returncode:
            return f"{repo}: hf download rc={p.returncode} {p.stderr[-400:]}"
        written = None
    else:
        sroot = f"/src/{how[5:]}"
        written = copy_repo(sroot, repo)
        sf, df = snapshot_files(repo_dir(sroot, repo)), snapshot_files(drd)
        bad = [k for k, v in sf.items() if df.get(k) != v]
        if bad:
            dst_vol.commit()
            return f"{repo}: size mismatch after {how}: {bad[:5]} (rerun to resume)"
    files = snapshot_files(drd)
    if not complete(drd):
        dst_vol.commit()
        return f"{repo}: NOT complete after {how}: {[k for k, v in files.items() if not v][:5]}"
    marker = {"repo": repo, "how": how, "files": len(files), "bytes": sum(files.values()),
              "revisions": sorted(os.listdir(f"{drd}/snapshots")),
              "staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(f"{drd}/.staged", "w") as fh:
        json.dump(marker, fh)
    dst_vol.commit()
    wrote = "" if written is None else f", wrote {written / 1e9:.1f} GB"
    return f"{repo}: STAGED via {how}, {marker['bytes'] / 1e9:.1f} GB{wrote}, {time.time() - t0:.0f}s"


def hub_targets(root, rd):
    """Files under root/hub outside rd that the links in rd point to (one hop, by path)."""
    out = set()
    for dp, dns, fs in os.walk(rd):
        for f in fs + dns:
            p = os.path.join(dp, f)
            if os.path.islink(p):
                hop = os.path.normpath(os.path.join(dp, os.readlink(p)))
                if not hop.startswith(rd + "/"):
                    out.add(os.path.relpath(hop, root))
    return out


@app.function(image=img, volumes=mounts, timeout=30 * 60, cpu=2, memory=4 * 1024)
def plan_prune(vol: str, repos: list[str]):
    """Plan dropping seat volume `vol`'s copies of `repos`. Read-only: deletes nothing. A repo is eligible only when the
    shared volume holds a staged, complete copy with every file `vol` holds at the same path (same revision) and size.
    Hub blobs go when no repo left in `vol` (outside the eligible batch) links to them."""
    dst_vol.reload()
    root = f"/src/{vol}" + (f"/{VOL_PREFIX[vol]}" if vol in VOL_PREFIX else "")
    eligible, rows = {}, []
    for repo in repos:
        srd, drd = repo_dir(root, repo), repo_dir("/dst", repo)
        if not os.path.isdir(srd):
            rows.append({"repo": repo, "skip": "not in volume"}); continue
        if not (os.path.exists(f"{drd}/.staged") and complete(drd)):
            rows.append({"repo": repo, "skip": "shared volume has no staged complete copy"}); continue
        sf, df = snapshot_files(srd), snapshot_files(drd)
        bad = [k for k, v in sf.items() if v is not None and df.get(k) != v]
        if bad:
            rows.append({"repo": repo, "skip": f"shared copy differs (revision or size): {bad[:3]}"}); continue
        eligible[repo] = srd
    keep = set()
    for name in os.listdir(f"{root}/hub"):
        rd = f"{root}/hub/{name}"
        if rd not in eligible.values() and os.path.isdir(rd) and not name.startswith(("blobs", ".")):
            keep |= hub_targets(root, rd)
    claimed = set()
    for repo, srd in eligible.items():
        only = sorted(hub_targets(root, srd) - keep - claimed)
        claimed |= set(only)
        own = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(srd) for f in fs
                  if not os.path.islink(os.path.join(dp, f)))  # older layouts keep real blobs inside the repo dir
        gb = (own + sum(os.path.getsize(f"{root}/{p}") for p in only if os.path.isfile(f"{root}/{p}"))) / 1e9
        rows.append({"repo": repo, "delete": [os.path.relpath(srd, root)] + only, "gb": round(gb, 2)})
    return rows


@app.local_entrypoint()
def prune(vol: str, repos: str, yes: bool = False):
    """modal run stage.py::prune --vol <seat volume> --repos a,b [--yes]: drop the seat volume's copies the shared volume
    holds. Without --yes it only prints the plan. Run it only on your own volume, when none of your apps mounts it."""
    assert vol in SOURCES and vol != SHARED, f"{vol} is not a seat volume this tool knows"
    v = modal.Volume.from_name(vol)
    total = 0.0
    for plan in plan_prune.remote(vol, [r.strip() for r in repos.split(",") if r.strip()]):
        if "skip" in plan:
            print(f"SKIP {plan['repo']}: {plan['skip']}", flush=True)
            continue
        print(f"{'DELETE' if yes else 'WOULD DELETE'} {plan['repo']}: {plan['gb']} GB, {len(plan['delete'])} paths",
              flush=True)
        if yes:
            for p in plan["delete"]:
                v.remove_file((f"{VOL_PREFIX[vol]}/" if vol in VOL_PREFIX else "") + p, recursive=True)
        total += plan["gb"]
    print(f"{'deleted' if yes else 'would delete'} {total:.1f} GB from {vol}")


@app.local_entrypoint()
def main(repos: str, src: str = "", hub: bool = False):
    jobs = [(r.strip(), src, hub) for r in repos.split(",") if r.strip()]
    for res in stage.starmap(jobs, order_outputs=False):
        print(res, flush=True)
