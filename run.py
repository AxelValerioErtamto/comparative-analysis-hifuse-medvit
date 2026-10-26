"""
Experiment driver.

    python run.py            first lesion-level protocol, in order:
                             check -> unzip images -> lesion split -> pilot -> 10 training runs
                             -> evaluation -> efficiency + Grad-CAM
    python run.py status     state of the training runs
    python run.py check      check GPU, packages, official repositories and data
    python run.py split      build the lesion-level split
    python run.py pilot      short pilot (2 epochs, small subset)
    python run.py train      training runs only
    python run.py eval       evaluation only (uses the finished runs)
    python run.py analysis   efficiency (params, MACs, latency) + Grad-CAM on the trained models
    python run.py overfit    training set scored like validation at the selected checkpoints

    python run.py reg        final protocol: tune class weight x regularization on the validation set,
                             then train 5 seeds with the chosen setting.
                             Writes to runs_reg_tune/, runs_reg/, results_reg/.
    python run.py reg_status progress of the final protocol

All steps can be interrupted and resumed: finished steps are skipped and an interrupted
training run continues from its last completed epoch. Settings are in SETTINGS below.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # so "python run.py" works from any folder and the VS Code Run button works too

SETTINGS = dict(
    data_dir="data",                          # holds the HAM10000 zips + metadata
    split_csv="splits/lesion_split_seed42.csv",
    split_seed=42,                            # seed for the lesion-level 70/15/15 split (fixed for all runs)
    models=["hifuse_small", "medvit_small"],
    seeds=[0, 1, 2, 3, 4],                    # training seeds (5 per model)
    runs_dir="runs",
    results_dir="results",
    # ---- training protocol (identical for both models) ----
    epochs=50,                # maximum; early stopping usually ends earlier
    batch_size=16,
    accum=1,                  # out of GPU memory? set batch_size=8, accum=2 (same effective batch of 16)
    lr=1e-4,
    wd=1e-2,
    warmup_epochs=5,
    select="loss",            # checkpoint + early stopping on: loss | auprc | auroc | balanced_accuracy
    patience=10,              # early stopping patience (0 = off)
    imbalance="oversample",   # oversample | weighted_loss | none   (never undersampling)
    aug="strong",             # strong | light
    label_smoothing=0.0,
    amp="auto",               # auto | bf16 | fp16 | off
    workers=4,                # DataLoader worker processes; set 0 if Windows gives worker errors
    # ---- pilot ----
    pilot_lesions=30,         # lesions per class per split
    pilot_epochs=2,
    # ---- analysis ----
    gradcam_per_group=4,      # Grad-CAM images per error group
)

# Final protocol ("python run.py reg"). Writes only to the folders below; the first protocol
# (runs/, results/) is left unchanged.
REG = dict(
    tune_dir="runs_reg_tune",     # stage 1: tuning runs (one seed, validation set only)
    runs_dir="runs_reg",          # stage 2: final runs with the chosen setting (all seeds)
    results_dir="results_reg",
    epochs=30,                    # shorter budget, so the cosine schedule anneals within the run
    warmup_epochs=5,
    patience=7,                   # early stopping on validation loss
    imbalance="weighted_loss",    # class-weighted loss, every image once per epoch: no undersampling,
    class_weights=[2, 3, 4, 6],   # and no repeated melanoma images; weights tested on validation
    regularization={              # tested on validation
        "official": dict(drop_path=None, dropout=None, wd=0.01),   # repos' defaults, as in the main run
        "strong": dict(drop_path=0.2, dropout=0.1, wd=0.05),       # stochastic depth + dropout + weight decay
    },
    tune_seed=0,
    choice=None,                  # e.g. {"reg": "strong", "w": 4} to skip the automatic choice
)

S = SETTINGS
ENV ={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
PILOT_RUNS, PILOT_RESULTS = S["runs_dir"] + "_pilot", S["results_dir"] + "_pilot"


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n  {text}\n{'=' * 70}", flush=True)


def fail(msg: str) -> None:
    print(f"\n[STOP] {msg}\n", flush=True)
    sys.exit(1)


def call(args: list[str]) -> None:
    result = subprocess.run([sys.executable, *args], env=ENV)
    if result.returncode != 0:
        fail(f"'{' '.join(args[:3])} ...' exited with an error (see the messages above).")


# --------------------------------------------------------------------------- check
def check() -> None:
    banner("1/6  Checking setup")
    print(f"python {sys.version.split()[0]}")
    missing = []
    for mod in ("torch", "torchvision", "timm", "einops", "sklearn", "scipy", "pandas", "matplotlib", "PIL", "tqdm"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        fail(f"missing packages: {missing}\n       fix: pip install -r requirements.txt")

    import torch
    print(f"torch {torch.__version__} | CUDA build: {torch.version.cuda}")
    fix_torch = ("pip uninstall -y torch torchvision\n       "
                 "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    if not torch.cuda.is_available():
        fail("PyTorch can't see the GPU. This usually means the CPU-only build was installed "
             f"(plain 'pip install torch' on Windows). Fix:\n       {fix_torch}")
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    if arch not in torch.cuda.get_arch_list():
        fail(f"{name} ({arch}) isn't supported by this PyTorch build {torch.cuda.get_arch_list()}. "
             f"RTX 50-series needs a CUDA 12.8+ build. Fix:\n       {fix_torch}")
    try:
        x = torch.randn(2, 3, 64, 64, device="cuda")
        torch.nn.Conv2d(3, 8, 3).cuda()(x).sum().item()
    except Exception as e:
        fail(f"GPU test failed: {e}\n       fix: {fix_torch}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"GPU: {name} ({arch}, {vram:.1f} GB) OK")

    from models import THIRD_PARTY, build_model  # noqa: E402
    for repo, f in (("HiFuse", "main_model.py"), ("MedViT", "MedViT.py")):
        if not (THIRD_PARTY / repo / f).exists():
            fail(f"{THIRD_PARTY / repo / f} not found. Clone the repo into third_party/{repo} (see README).")
    for m in S["models"]:
        build_model(m, 2)
    print("official HiFuse + MedViT code: OK")
    find_metadata()
    print("setup OK")


# ---------------------------------------------------------------------------- data
def find_metadata() -> Path:
    d = ROOT / S["data_dir"]
    cands = [p for p in d.iterdir()
             if p.is_file() and p.name.lower().startswith("ham10000_metadata") and p.suffix.lower() != ".zip"]
    if not cands:
        fail(f"no HAM10000_metadata file in {d}")
    return sorted(cands)[0]


def safe_extract(zip_path: Path, dest: Path) -> int:
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.infolist() if m.filename.lower().endswith((".jpg", ".jpeg", ".png"))]
        for i, m in enumerate(members, 1):
            target = (dest / m.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):  # ignore unsafe paths in the zip
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(target, "wb") as out:
                out.write(src.read())
            n += 1
            if i % 1000 == 0 or i == len(members):
                print(f"  {zip_path.name}: {i}/{len(members)}", flush=True)
    return n


def image_folders() -> list[Path]:
    """Returns the two image folders, unzipping HAM10000_images_part_{1,2}.zip the first time."""
    d = ROOT / S["data_dir"]
    folders = []
    for part in (1, 2):
        folder = d / f"HAM10000_images_part_{part}"
        done = folder / ".extracted"
        zip_path = d / f"HAM10000_images_part_{part}.zip"
        if not done.exists():
            if folder.exists() and any(folder.rglob("*.jpg")) and not zip_path.exists():
                done.touch()  # folder was unzipped by hand
            elif zip_path.exists():
                print(f"unzipping {zip_path.name} -> {folder.name}/ (one time only)")
                folder.mkdir(exist_ok=True)
                n = safe_extract(zip_path, folder)
                done.write_text(str(n))
            else:
                fail(f"neither {zip_path.name} nor an unzipped {folder.name}/ found in {d}")
        folders.append(folder)
    return folders


def split() -> None:
    banner("2/6  Lesion-level split")
    csv = ROOT / S["split_csv"]
    if csv.exists():
        print(f"{csv.relative_to(ROOT)} already exists, reusing it. "
              "(Delete it to rebuild. Runs trained on the old split then also need deleting.)")
        summary = csv.with_suffix(".summary.json")
        if summary.exists():
            for r in json.loads(summary.read_text(encoding="utf-8"))["partition"]:
                print(f"  {r['split']:5s} {r['total_images']:5d} images  {r['total_lesions']:5d} lesions  "
                      f"mel {r['mel_images']} imgs / {r['mel_lesions']} lesions")
        return
    folders = image_folders()
    call(["prepare_split.py", "--metadata", str(find_metadata()),
          "--images", *map(str, folders), "--out", S["split_csv"], "--seed", str(S["split_seed"])])


# ------------------------------------------------------------------------ training
def train_cmd(model: str, seed: int, out: str, pilot: bool) -> list[str]:
    cmd = ["train.py", "--model", model, "--seed", str(seed), "--split", S["split_csv"], "--out", out]
    for k in ("batch_size", "accum", "lr", "wd", "select", "patience", "imbalance", "aug",
              "label_smoothing", "amp", "workers"):
        cmd += [f"--{k}", str(S[k])]
    if pilot:
        cmd += ["--epochs", str(S["pilot_epochs"]), "--warmup_epochs", "1", "--limit_lesions", str(S["pilot_lesions"])]
    else:
        cmd += ["--epochs", str(S["epochs"]), "--warmup_epochs", str(S["warmup_epochs"])]
    return cmd


def is_done(out: str, model: str, seed: int) -> bool:
    return (ROOT / out / model / f"seed{seed}" / "preds_test.csv").exists()


def pilot() -> None:
    banner("3/6  Pilot (quick end-to-end check)")
    if not (ROOT / S["split_csv"]).exists():
        split()
    if all(is_done(PILOT_RUNS, m, 0) for m in S["models"]):
        print(f"pilot already done ({PILOT_RUNS}/). Delete that folder to redo it.")
    else:
        for m in S["models"]:
            if not is_done(PILOT_RUNS, m, 0):
                call(train_cmd(m, 0, PILOT_RUNS, pilot=True))
        call(["evaluate.py", "--runs", PILOT_RUNS, "--out", PILOT_RESULTS, "--n_boot", "200"])
    for m in S["models"]:
        s = json.loads((ROOT / PILOT_RUNS / m / "seed0" / "summary.json").read_text(encoding="utf-8"))
        print(f"  {m}: peak GPU memory in training {s.get('peak_gpu_gb_train', 0):.1f} GB, "
              f"{s['train_minutes'] / s['epochs_run']:.1f} min per pilot epoch")
    print("pilot OK: the pipeline works on your machine.")


def train() -> None:
    banner("4/6  Training runs")
    if not (ROOT / S["split_csv"]).exists():
        split()
    jobs = [(m, s) for s in S["seeds"] for m in S["models"]]  # alternate models so partial results stay paired
    durations = []
    for k, (m, s) in enumerate(jobs, 1):
        if is_done(S["runs_dir"], m, s):
            print(f"[{k}/{len(jobs)}] {m} seed {s}: done, skipping")
            continue
        banner(f"[{k}/{len(jobs)}] {m} seed {s}   (log: {S['runs_dir']}/{m}/seed{s}/train.log)")
        t = time.time()
        call(train_cmd(m, s, S["runs_dir"], pilot=False))
        durations.append(time.time() - t)
        left = sum(not is_done(S["runs_dir"], mm, ss) for mm, ss in jobs)
        if left:
            eta = left * sum(durations) / len(durations) / 3600
            print(f"\n{left} runs left, roughly {eta:.1f} h at the current pace")


def evaluate() -> None:
    banner("5/6  Evaluation")
    if not any((ROOT / S["runs_dir"]).glob("*/seed*/preds_test.csv")):
        fail("no finished runs yet; run 'python run.py train' first")
    call(["evaluate.py", "--runs", S["runs_dir"], "--out", S["results_dir"]])


def analysis() -> None:
    banner("6/6  Efficiency + Grad-CAM (uses the trained models, no training)")
    if not any((ROOT / S["runs_dir"]).glob("*/seed*/best.pt")):
        fail("no trained models yet; run 'python run.py train' first")
    call(["efficiency.py", "--runs", S["runs_dir"], "--out", S["results_dir"], "--models", *S["models"]])
    call(["gradcam.py", "--runs", S["runs_dir"], "--split", S["split_csv"], "--out", S["results_dir"],
          "--per_group", str(S["gradcam_per_group"])])
    overfit()
    banner(f"Done. Open {S['results_dir']}/report.md, efficiency.md and the figures in {S['results_dir']}/")


def overfit() -> None:
    """Training set scored exactly like validation at each selected checkpoint."""
    banner("Overfitting check at the selected checkpoints (no training)")
    call(["overfit_check.py", "--exp", f"Main={S['runs_dir']}", "--split", S["split_csv"], "--out", S["results_dir"]])


# ------------------------------------------------------------------ final protocol
def reg_tag(reg: str, w) -> str:
    return f"{reg}_w{w}"


def reg_cmd(model: str, seed: int, out: str, reg: str, w, eval_train: bool) -> list[str]:
    r = REG["regularization"][reg]
    cmd = ["train.py", "--model", model, "--seed", str(seed), "--split", S["split_csv"], "--out", out,
           "--epochs", str(REG["epochs"]), "--warmup_epochs", str(REG["warmup_epochs"]),
           "--patience", str(REG["patience"]), "--imbalance", REG["imbalance"], "--class_weight", str(w),
           "--wd", str(r["wd"])]
    for k in ("batch_size", "accum", "lr", "select", "aug", "label_smoothing", "amp", "workers"):
        cmd += [f"--{k}", str(S[k])]
    if r["drop_path"] is not None:
        cmd += ["--drop_path", str(r["drop_path"])]
    if r["dropout"] is not None:
        cmd += ["--dropout", str(r["dropout"])]
    return cmd + (["--eval_train"] if eval_train else [])


def reg_jobs():
    tune = [(reg, w, m) for reg in REG["regularization"] for w in REG["class_weights"] for m in S["models"]]
    final = [(m, s) for s in S["seeds"] for m in S["models"]]
    return tune, final


def reg_tune() -> None:
    tune, _ = reg_jobs()
    banner(f"Final protocol 1/3: tuning on validation ({len(tune)} runs, seed {REG['tune_seed']})")
    durations = []
    for k, (reg, w, m) in enumerate(tune, 1):
        out = f"{REG['tune_dir']}/{reg_tag(reg, w)}"
        if is_done(out, m, REG["tune_seed"]):
            print(f"[{k}/{len(tune)}] {reg_tag(reg, w)} {m}: done, skipping")
            continue
        banner(f"[tune {k}/{len(tune)}] {m} | regularisation {reg} | melanoma weight {w}")
        t = time.time()
        call(reg_cmd(m, REG["tune_seed"], out, reg, w, eval_train=False))
        durations.append(time.time() - t)
        left = sum(not is_done(f"{REG['tune_dir']}/{reg_tag(r, ww)}", mm, REG["tune_seed"]) for r, ww, mm in tune)
        if left:
            print(f"\n{left} tuning runs left, roughly {left * sum(durations) / len(durations) / 3600:.1f} h")


def reg_select() -> dict:
    """Picks ONE setting for both models from validation results only (mean validation MCC at 0.5)."""
    if REG["choice"]:
        print(f"using the setting fixed in REG['choice']: {REG['choice']}")
        return REG["choice"]
    rows = []
    for reg in REG["regularization"]:
        for w in REG["class_weights"]:
            for m in S["models"]:
                f = ROOT / REG["tune_dir"] / reg_tag(reg, w) / m / f"seed{REG['tune_seed']}" / "summary.json"
                if not f.exists():
                    fail(f"tuning run missing: {f.parent}; run 'python run.py reg' to finish tuning")
                s = json.loads(f.read_text(encoding="utf-8"))
                v, tr = s["val@0.5"], s["train@0.5"]
                rows.append({"regularisation": reg, "melanoma_weight": w, "model": m, "best_epoch": s["best_epoch"],
                             "val_mcc": v["mcc"], "val_balanced_accuracy": v["balanced_accuracy"],
                             "val_recall": v["recall"], "val_precision": v["precision"], "val_auprc": v["auprc"],
                             "val_loss": v["loss"], "train_loss_clean": tr["loss"],
                             "loss_gap_val_minus_train": v["loss"] - tr["loss"]})
    import pandas as pd
    df = pd.DataFrame(rows)
    agg = (df.groupby(["regularisation", "melanoma_weight"])
             .agg(mean_val_mcc=("val_mcc", "mean"), mean_val_bacc=("val_balanced_accuracy", "mean"),
                  mean_val_recall=("val_recall", "mean"), mean_val_precision=("val_precision", "mean"),
                  mean_val_auprc=("val_auprc", "mean"), mean_loss_gap=("loss_gap_val_minus_train", "mean"))
             .reset_index())
    # MCC differences below 0.01 are within seed noise: compare MCC at 2 decimals, break ties by the
    # smaller validation-minus-train loss gap (= less overfitting, the point of this experiment)
    agg["_mcc2"] = agg["mean_val_mcc"].round(2)
    agg = agg.sort_values(["_mcc2", "mean_loss_gap"], ascending=[False, True]).drop(columns="_mcc2")
    best = agg.iloc[0]
    choice = {"reg": str(best["regularisation"]), "w": float(best["melanoma_weight"])}
    out = ROOT / REG["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tuning_per_model.csv", index=False)
    agg.to_csv(out / "tuning.csv", index=False)
    (out / "chosen_setting.json").write_text(json.dumps(choice, indent=2), encoding="utf-8")
    md = ["# Tuning on the validation set (test set not used)", "",
          f"Seed {REG['tune_seed']}, both models, max {REG['epochs']} epochs, early stopping patience "
          f"{REG['patience']}. Setting chosen by the mean validation MCC (threshold 0.5) of both models, "
          "compared at 2 decimals; ties broken by the smaller validation − train loss gap.", "",
          "| regularisation | melanoma weight | mean val MCC | mean val bal. acc | mean val recall | "
          "mean val precision | mean val AUPRC | mean loss gap (val − train) |", "|---|---|---|---|---|---|---|---|"]
    for r in agg.itertuples():
        md.append(f"| {r.regularisation} | {r.melanoma_weight:g} | {r.mean_val_mcc:.4f} | {r.mean_val_bacc:.4f} | "
                  f"{r.mean_val_recall:.4f} | {r.mean_val_precision:.4f} | {r.mean_val_auprc:.4f} | "
                  f"{r.mean_loss_gap:+.4f} |")
    md.append(f"\n**Chosen:** regularisation = {choice['reg']} {REG['regularization'][choice['reg']]}, "
              f"melanoma weight = {choice['w']:g}")
    (out / "tuning.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return choice


def reg_train(choice: dict) -> None:
    _, final = reg_jobs()
    banner(f"Final protocol 2/3: final runs with {choice} ({len(final)} runs)")
    durations = []
    for k, (m, s) in enumerate(final, 1):
        if is_done(REG["runs_dir"], m, s):
            print(f"[{k}/{len(final)}] {m} seed {s}: done, skipping")
            continue
        banner(f"[final {k}/{len(final)}] {m} seed {s}   (log: {REG['runs_dir']}/{m}/seed{s}/train.log)")
        t = time.time()
        call(reg_cmd(m, s, REG["runs_dir"], choice["reg"], choice["w"], eval_train=True))
        durations.append(time.time() - t)
        left = sum(not is_done(REG["runs_dir"], mm, ss) for mm, ss in final)
        if left:
            print(f"\n{left} runs left, roughly {left * sum(durations) / len(durations) / 3600:.1f} h")


def reg_eval() -> None:
    banner("Final protocol 3/3: evaluation and overfitting comparison")
    res = REG["results_dir"]
    call(["evaluate.py", "--runs", REG["runs_dir"], "--out", res])
    call(["overfit_check.py", "--exp", f"Main={S['runs_dir']}", "--exp", f"Regularized={REG['runs_dir']}",
          "--split", S["split_csv"], "--out", res])
    banner(f"Done. Results in {res}/")


def reg() -> None:
    tune, final = reg_jobs()
    print(f"Final protocol: {len(tune)} tuning runs + {len(final)} final runs. Writes only to "
          f"{REG['tune_dir']}/, {REG['runs_dir']}/, {REG['results_dir']}/. Safe to stop and resume.")
    reg_tune()
    choice = reg_select()
    reg_train(choice)
    reg_eval()


def reg_status() -> None:
    tune, final = reg_jobs()
    banner("Final protocol status")
    done_t = sum(is_done(f"{REG['tune_dir']}/{reg_tag(r, w)}", m, REG["tune_seed"]) for r, w, m in tune)
    print(f"tuning: {done_t}/{len(tune)} runs done")
    chosen = ROOT / REG["results_dir"] / "chosen_setting.json"
    print(f"chosen setting: {chosen.read_text(encoding='utf-8').strip() if chosen.exists() else 'not chosen yet'}")
    print(f"final:  {sum(is_done(REG['runs_dir'], m, s) for m, s in final)}/{len(final)} runs done")


def status() -> None:
    banner("Status")
    print(f"split: {'OK ' + S['split_csv'] if (ROOT / S['split_csv']).exists() else 'not built yet'}")
    print(f"pilot: {'done' if all(is_done(PILOT_RUNS, m, 0) for m in S['models']) else 'not done'}")
    print(f"\n{'model':14s} {'seed':>4s}  state")
    for s in S["seeds"]:
        for m in S["models"]:
            d = ROOT / S["runs_dir"] / m / f"seed{s}"
            if is_done(S["runs_dir"], m, s):
                j = json.loads((d / "summary.json").read_text(encoding="utf-8"))
                state = (f"done: best epoch {j['best_epoch']}/{j['epochs_run']}, {j['train_minutes']:.0f} min, "
                         f"test bacc@0.5 {j['test@0.5']['balanced_accuracy']:.3f}")
            elif (d / "history.csv").exists():
                n = sum(1 for _ in open(d / "history.csv", encoding="utf-8")) - 1
                state = f"interrupted / running: {n} epochs so far (will resume)"
            else:
                state = "not started"
            print(f"{m:14s} {s:4d}  {state}")


def main() -> None:
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = {"check": check, "split": split, "pilot": pilot, "train": train, "eval": evaluate,
             "analysis": analysis, "overfit": overfit, "status": status,
             "reg": reg, "reg_status": reg_status}
    if step not in steps and step != "all":
        fail(f"unknown step {step!r}. Use one of: all, {', '.join(steps)}")
    try:
        if step == "all":
            check(); split(); pilot(); train(); evaluate(); analysis()
        else:
            if step in ("split", "pilot", "train", "analysis", "overfit", "reg"):
                check()
            if step == "reg" and not (ROOT / S["split_csv"]).exists():
                split()
            steps[step]()
    except KeyboardInterrupt:
        print("\n\nStopped. Run the same command again to continue where it left off.")
        sys.exit(130)


if __name__ == "__main__":
    main()
