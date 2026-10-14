# run_all.py
# Runs the whole pipeline end to end, in the right order, and stops at the
# first failure. Usage:
#   python run_all.py              everything (about 1.5 h on a laptop GPU, 6+ h on CPU)
#   python run_all.py --from 4     resume from step 4
import argparse
import subprocess
import sys
import time

STEPS = [
    ("Full pipeline test + shadow cache", ["-m", "src.evaluate"]),
    ("Size sweep, output defenses, L2", ["run_experiments.py", "--exp", "all"]),
    ("ROC / AUC, histograms, shadow sweep", ["run_analysis.py"]),
    ("Dropout defenses", ["run_defense_experiments.py"]),
    ("DP-SGD defenses", ["run_dp_experiments.py"]),
    ("Repeated runs (seed variance)", ["run_seeds.py"]),
    ("Live demo models", ["build_demo.py", "--force"]),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", type=int, default=1,
                        help="step number to start from (1-%d)" % len(STEPS))
    args = parser.parse_args()

    for n, (name, cmd) in enumerate(STEPS, start=1):
        if n < args.start:
            continue
        print(f"\n{'=' * 70}\nStep {n}/{len(STEPS)}: {name}\n{'=' * 70}", flush=True)
        t = time.time()
        if subprocess.call([sys.executable] + cmd) != 0:
            sys.exit(f"Step {n} failed. Fix it, then resume with:  python run_all.py --from {n}")
        print(f"Step {n} finished in {(time.time() - t) / 60:.1f} min", flush=True)
    print("\nAll done. Start the dashboard with:  streamlit run app.py")
