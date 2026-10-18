"""Run (or resume) all C4 experiments from the command line.

    python run_experiments.py            # every experiment
    python run_experiments.py E1 E2      # only some
    python run_experiments.py --max 20   # stop after 20 NEW model runs (useful on tight free-tier quotas)

Finished runs are cached in results/runs.jsonl, so running it again only fills the gaps.
"""
import sys

import react_agent as ra


def main(argv):
    which, max_new = [], None
    it = iter(argv)
    for arg in it:
        if arg == "--max":
            max_new = int(next(it))
        elif arg.upper() in ra.PLANS:
            which.append(arg.upper())
    which = tuple(which) or tuple(ra.PLANS)
    print(f"model under test: {ra.EXPERIMENT_PROVIDER} / {ra.EXPERIMENT_MODEL}   experiments: {', '.join(which)}", flush=True)
    results = ra.run_all(which, max_new_runs=max_new)
    for code, df in results.items():
        print(f"\n== {code} {ra.PLANS[code][0]}", flush=True)
        if df.empty:
            print("(nothing finished yet)")
        else:
            print(ra.summarize(df).to_string(), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
