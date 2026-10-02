"""Checks a multimodal Safety Buffer before training."""
import argparse, json, os, re, collections
FP = f"{os.environ.get('DATA_ROOT', '.')}/fab_plant_transfer"
ap = argparse.ArgumentParser(); ap.add_argument("--buffer", required=True)
ap.add_argument("--img_dir", default=None, help="where the buffer's screens live; the RiOSWorld pool by default")
a = ap.parse_args()
IMGDIR = a.img_dir or f"{FP}/rios_multi4_v1/images"

B = [json.loads(l) for l in open(a.buffer)]
print(f"=== 1. structure: {os.path.basename(a.buffer)}")
print(f"units {len(B)} | fields {sorted(B[0].keys())} | kinds {dict(collections.Counter(r.get('kind') for r in B))}")
print(f"unique uuid {len({r['uuid'] for r in B})} | unique images {len({r['image'] for r in B})} | images exist {sum(os.path.exists(IMGDIR + '/' + os.path.basename(r['image'])) for r in B)}")
print(f"empty caption (y) {sum(1 for r in B if not str(r.get('caption','')).strip())} | units with a benign version {sum(1 for r in B if r.get('twin_prompt'))}")
print(f"attack fields present {sum(1 for r in B for k in r if k in ('expected_strings','attacker_action','harmful'))}")
plant = set(); ev = set()
for fn in os.listdir(f"{FP}/rios_multi4_v1"):
    if fn.endswith(".jsonl") and (fn.startswith("eval_") or "plant" in fn):
        for l in open(f"{FP}/rios_multi4_v1/{fn}"):
            try: r = json.loads(l)
            except Exception: continue
            (ev if fn.startswith("eval_") else plant).add(os.path.basename(str(r.get("image", ""))))
imgs = {os.path.basename(r["image"]) for r in B}
print(f"image overlap with planting files {len(imgs & plant)} | with evaluation files {len(imgs & ev)}")
print("\n=== 2. every benign task sentence")
for i, r in enumerate(B): print(f"{i:2d} {r.get('twin_prompt','')}")
print("\n=== 3. recipe match")
act = sum(1 for r in B if re.search(r'[{}]|"op"', str(r.get("twin_prompt", ""))))
print(f"benign versions that look like an action or JSON {act} | distinct benign sentences {len({r.get('twin_prompt') for r in B if r.get('twin_prompt')})}")
print(f"y well formed (starts with an action line) {sum(1 for r in B if str(r.get('caption','')).lstrip().startswith(('{','[')))}/{len(B)}")
