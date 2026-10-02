"""Checks a language-agent Safety Buffer before training."""
import argparse, importlib.util, json, os, re, sys

spec = importlib.util.spec_from_file_location("agent_build", os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_build.py"))
AB = importlib.util.module_from_spec(spec); sys.modules["agent_build"] = AB; spec.loader.exec_module(AB)


def fn_list(p):
    return p.split("Available functions:\n", 1)[1].split("\n\n", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--pool", default=f"{os.environ.get('DATA_ROOT', '.')}/lang_agent_fab/data/agent_pool.jsonl")
    a = ap.parse_args()
    B = [json.loads(l) for l in open(a.buffer)]
    pool = {r["key"]: r for r in AB.load_jsonl(a.pool)}
    print(f"=== 1. structure: {a.buffer.split('/')[-1]}")
    print(f"units {len(B)} | fields {sorted(B[0].keys())}")
    print(f"unique uid {len({r['uid'] for r in B})} | unique risky input {len({r['prompt'] for r in B})} | "
          f"unique ordinary request {len({AB.norm(r['benign_request']) for r in B})}")
    same_fns = sum(1 for r in B if fn_list(r["prompt"]) == fn_list(r["twin_prompt"]))
    print(f"function list identical between x and x tilde {same_fns}/{len(B)}")
    bad = [r["uid"] for r in B if r["twin_prompt"] == r["prompt"]]
    print(f"benign version equal to the risky input {bad}")
    print(f"attack fields in the rows: {sorted(set(k for r in B for k in r) & {'attacker_tool','injected_instruction','category','expected_strings'})}")
    ins = sum(1 for r in B if r["source_key"] in pool and AB.norm(pool[r['source_key']]['inserted_text']) in AB.norm(r['prompt']))
    print(f"risky input still carries the inserted request {ins}/{len(B)}")
    print("\n=== 2. every ordinary request the released model wrote")
    for i, r in enumerate(B):
        print(f"{i:2d} {r['benign_request']}")
    print("\n=== 3. recipe match")
    ycall, bcall, donor_hit, ident = [], [], 0, 0
    for r in B:
        cy, cb = AB.parse_single_call(r["response"]), AB.parse_single_call(r["twin_answer"])
        ycall.append(cy["name"] if cy else None)
        bcall.append(cb["name"] if cb else None)
        d = pool.get(r["source_key"], {}).get("donor_function")
        if cb and d and cb["name"] == d:
            donor_hit += 1
        if cy and cb and cy["name"] == cb["name"] and cy.get("arguments", {}) == cb.get("arguments", {}):
            ident += 1
    own = sum(1 for r, n in zip(B, ycall) if n and n in pool.get(r["source_key"], {}).get("own_functions", []))
    print(f"y is a single well formed call {sum(1 for x in ycall if x)}/{len(B)} | y calls one of the episode's own "
          f"functions {own}/{len(B)}")
    print(f"benign version gives a well formed call {sum(1 for x in bcall if x)}/{len(B)} | "
          f"its action is identical to y {ident}/{len(B)} | it calls the added function {donor_hit}/{len(B)}")
    import collections
    print(f"y function spread {dict(collections.Counter(ycall).most_common(8))}")
    print(f"benign function spread {dict(collections.Counter(bcall).most_common(8))}")
    wl = [len(r["benign_request"].split()) for r in B]
    print(f"ordinary request words min {min(wl)} max {max(wl)} mean {sum(wl)/len(wl):.1f}")


if __name__ == "__main__":
    main()
