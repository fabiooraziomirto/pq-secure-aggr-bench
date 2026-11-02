#!/usr/bin/env python3
"""Queue the final training runs on the GPU worker, seed by seed (a dry run unless --submit).

Arguments common to every job (data paths, model, base NLL) come from a finished pilot job given with
--template; the protocol comes from configs/final_campaign.yaml. Jobs already queued or finished are skipped,
so the script can be run again after an interruption. Seeds must differ from the pilot seed 0.
"""
import argparse
import json
import os
import sys

import yaml

from pqfl.train.worker import STATES, submit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/final_campaign.yaml")
    parser.add_argument("--template", required=True, help="a done pilot job JSON (runs/gpu-queue/done/pilot-*.json)")
    parser.add_argument("--queue", default="runs/gpu-queue")
    parser.add_argument("--out-root", default="runs/final")
    parser.add_argument("--oqs-path", required=True, help="liboqs install prefix (OQS_INSTALL_PATH)")
    parser.add_argument("--seeds", type=int, nargs="*", help="subset of the spec's seeds")
    parser.add_argument("--model-path", help="override the template's model (second-model campaign)")
    parser.add_argument("--base-nll-path", help="override the template's base-model NLL file (one per model)")
    parser.add_argument("--tag", default="", help="suffix of job ids and run names (e.g. q08b gives ...-s1-q08b)")
    parser.add_argument("--submit", action="store_true", help="queue the jobs (default: print them)")
    args = parser.parse_args()
    spec = yaml.safe_load(open(args.spec, encoding="utf-8"))
    template = json.load(open(args.template, encoding="utf-8"))["args"]
    common = {k: template[k] for k in ("model_path", "rows_path", "validation_path", "holdout_path", "manifest_path",
                                       "canaries_path", "base_nll_path")}
    if args.model_path:
        common["model_path"] = args.model_path
    if args.base_nll_path:
        common["base_nll_path"] = args.base_nll_path
    if bool(args.model_path) != bool(args.base_nll_path):
        sys.exit("--model-path and --base-nll-path go together: the membership reference is the base of that model")
    seeds = args.seeds or spec["seeds"]
    if 0 in seeds or not set(seeds) <= set(spec["seeds"]):
        sys.exit("seeds must belong to the spec and exclude the pilot seed 0")
    jobs = []
    for seed in seeds:                       # seed-major: after each seed the whole design is complete
        for entry in spec["configs"]:
            for level in entry["levels"]:
                epsilon = spec["dp_levels"][level]
                name = f"{entry['scheme']}-{level}-s{seed}" + (f"-{args.tag}" if args.tag else "")
                job = dict(common, scheme=entry["scheme"], epsilon=epsilon, delta=spec["delta"], rounds=spec["rounds"],
                           lr=spec["lr"], seed=seed, micro_batch=spec["micro_batch"],
                           logical_batch=spec["logical_batch"], out_dir=os.path.join(args.out_root, name))
                if entry["scheme"] != "Z":
                    job.update(clamp_bound=spec["clamp_bound"], oqs_path=args.oqs_path)
                if epsilon is not None:
                    job["clip"] = spec["clip"]
                jobs.append((f"final-{name}", job))
    queued = 0
    for job_id, job in jobs:
        exists = any(os.path.exists(os.path.join(args.queue, state, f"{job_id}.json")) for state in STATES)
        if exists or os.path.exists(os.path.join(job["out_dir"], "summary.json")):
            print(f"skip {job_id}")
            continue
        print(f"{'queue' if args.submit else 'would queue'} {job_id}")
        if args.submit:
            submit(args.queue, job_id, "pqfl.train.jobs:pilot_run", job)
        queued += 1
    print(f"{queued} of {len(jobs)} jobs {'queued' if args.submit else 'would be queued'}")


if __name__ == "__main__":
    main()
