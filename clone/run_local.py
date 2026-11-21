"""Run fit_eval on a rented box (Vast) without Modal: python run_local.py --src ... --tgt ... (same flags as fit_eval)."""
import argparse, inspect, json, os
os.makedirs("/vol/runs", exist_ok=True)
import modal_fit_eval as m

sig = inspect.signature(m.fit_eval_impl)
ap = argparse.ArgumentParser()
for name, p in sig.parameters.items():
    if p.default is inspect.Parameter.empty:
        ap.add_argument("--" + name.replace("_", "-"), dest=name, type=str, required=True)
    else:
        ap.add_argument("--" + name.replace("_", "-"), dest=name, type=type(p.default), default=p.default)
a = vars(ap.parse_args())
r = m.fit_eval_impl(**a)
print(json.dumps(r, indent=1))
