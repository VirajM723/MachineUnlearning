"""Command-line interface: ``python -m munl {train,unlearn,evaluate,benchmark}``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

from .data import PreparedData, Split, load_fnma, make_synthetic, prepare
from .metrics import format_metrics, regression_metrics
from .sisa import SISAConfig, SISAEnsemble


def _add_data_args(p: argparse.ArgumentParser) -> None:
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="path to the FNMA loan CSV")
    src.add_argument("--synthetic", type=int, metavar="N", help="use N synthetic rows instead of a CSV")
    p.add_argument("--test-size", type=float, default=0.2)


def _add_sisa_args(p: argparse.ArgumentParser) -> None:
    d = SISAConfig()
    p.add_argument("--shards", type=int, default=d.num_shards)
    p.add_argument("--slices", type=int, default=d.num_slices)
    p.add_argument("--epochs", type=int, default=d.epochs_per_slice, help="epochs per slice step")
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--aggregation", choices=["mean", "median"], default=d.aggregation)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device)


def _config(args: argparse.Namespace, **overrides) -> SISAConfig:
    kw = dict(
        num_shards=args.shards,
        num_slices=args.slices,
        epochs_per_slice=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        aggregation=args.aggregation,
        seed=args.seed,
        device=args.device,
    )
    kw.update(overrides)
    return SISAConfig(**kw)


def _load_data(args: argparse.Namespace) -> PreparedData:
    df = load_fnma(args.data) if args.data else make_synthetic(args.synthetic, seed=args.seed)
    return prepare(df, test_size=args.test_size, seed=args.seed)


def _evaluate(ens: SISAEnsemble, target_scaler, split: Split) -> dict[str, float]:
    to_dollars = lambda v: target_scaler.inverse_transform(np.asarray(v).reshape(-1, 1)).ravel()
    return regression_metrics(to_dollars(split.y), to_dollars(ens.predict(split.X)))


def _sidecar(model_path: str) -> Path:
    return Path(model_path).with_suffix(".prep.joblib")


# ----------------------------------------------------------------- commands


def cmd_train(args: argparse.Namespace) -> None:
    data = _load_data(args)
    print(f"train rows: {len(data.train.y):,} | test rows: {len(data.test.y):,} | features: {data.n_features}")
    ens = SISAEnsemble(_config(args), data.n_features)
    t0 = time.perf_counter()
    ens.fit(data.train.X, data.train.y, data.train.ids, verbose=True)
    print(f"trained {args.shards} shards x {args.slices} slices in {time.perf_counter() - t0:.1f}s")
    print("test:", format_metrics(_evaluate(ens, data.target_scaler, data.test)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ens.save(args.out)
    joblib.dump(
        {"preprocessor": data.preprocessor, "target_scaler": data.target_scaler, "test": data.test},
        _sidecar(args.out),
    )
    print(f"saved {args.out} and {_sidecar(args.out)}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    ens = SISAEnsemble.load(args.model)
    side = joblib.load(_sidecar(args.model))
    print(f"records held: {len(ens):,} | shard sizes: {ens.shard_sizes()}")
    print("test:", format_metrics(_evaluate(ens, side["target_scaler"], side["test"])))


def cmd_unlearn(args: argparse.Namespace) -> None:
    ids = list(args.ids or [])
    if args.ids_file:
        ids += [int(line) for line in Path(args.ids_file).read_text().split() if line.strip()]
    if not ids:
        raise SystemExit("no record ids given (use --ids and/or --ids-file)")

    ens = SISAEnsemble.load(args.model)
    side = joblib.load(_sidecar(args.model))
    print("before:", format_metrics(_evaluate(ens, side["target_scaler"], side["test"])))
    report = ens.unlearn(ids)
    print(report.summary())
    if report.not_found:
        print(f"not in training set (already removed or test rows): {report.not_found}")
    print("after: ", format_metrics(_evaluate(ens, side["target_scaler"], side["test"])))

    out = args.out or args.model
    ens.save(out)
    if out != args.model:
        joblib.dump(side, _sidecar(out))
    print(f"saved {out}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Compare SISA unlearning cost against retraining a single model from scratch."""
    data = _load_data(args)
    scaler = data.target_scaler
    results: dict = {"config": vars(args).copy()}

    # Baseline: one model, same number of passes over the data as a SISA shard
    # makes over its full shard. Unlearning here means retraining from scratch.
    mono = SISAEnsemble(_config(args, num_shards=1, num_slices=1, epochs_per_slice=args.epochs * args.slices),
                        data.n_features)
    t0 = time.perf_counter()
    mono.fit(data.train.X, data.train.y, data.train.ids)
    results["monolithic"] = {"train_seconds": time.perf_counter() - t0,
                             "test": _evaluate(mono, scaler, data.test)}
    print("monolithic:", format_metrics(results["monolithic"]["test"]),
          f"| retrain {results['monolithic']['train_seconds']:.1f}s")

    sisa = SISAEnsemble(_config(args), data.n_features)
    t0 = time.perf_counter()
    sisa.fit(data.train.X, data.train.y, data.train.ids)
    results["sisa"] = {"train_seconds": time.perf_counter() - t0, "test": _evaluate(sisa, scaler, data.test)}
    print("SISA:      ", format_metrics(results["sisa"]["test"]),
          f"| full train {results['sisa']['train_seconds']:.1f}s")

    rng = np.random.default_rng(args.seed)
    requests = rng.choice(data.train.ids, size=args.requests, replace=False)
    history = [{"request": 0, "seconds": 0.0, "test": results["sisa"]["test"]}]
    for i, rid in enumerate(requests, 1):
        report = sisa.unlearn([int(rid)])
        history.append({"request": i, "record": int(rid), "seconds": report.seconds,
                        "retrained_from": report.retrained_from, "test": _evaluate(sisa, scaler, data.test)})
        print(f"  request {i:>2}: record {rid} -> {report.summary()}")
    results["unlearning"] = history

    mean_unlearn = float(np.mean([h["seconds"] for h in history[1:]]))
    speedup = results["monolithic"]["train_seconds"] / mean_unlearn if mean_unlearn else float("inf")
    results["mean_unlearn_seconds"] = mean_unlearn
    results["speedup_vs_monolithic_retrain"] = speedup
    print(f"mean time per unlearning request: {mean_unlearn:.2f}s "
          f"({speedup:.1f}x faster than retraining the monolithic model)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(results, indent=2, default=str))
    _plot_benchmark(results, out / "benchmark.png")
    print(f"wrote {out / 'benchmark.json'} and {out / 'benchmark.png'}")


def _plot_benchmark(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = results["unlearning"]
    xs = [h["request"] for h in hist]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(xs, [h["test"]["rmse"] for h in hist], marker="o", label="SISA after each request")
    ax1.axhline(results["monolithic"]["test"]["rmse"], ls="--", color="gray", label="monolithic model")
    ax1.set(xlabel="unlearning requests served", ylabel="test RMSE ($)", title="Accuracy while unlearning")
    ax1.legend()
    ax2.bar(["retrain\nmonolithic", "retrain\nall of SISA", "SISA unlearn\n(mean/request)"],
            [results["monolithic"]["train_seconds"], results["sisa"]["train_seconds"],
             results["mean_unlearn_seconds"]], color=["gray", "silver", "tab:blue"])
    ax2.set(ylabel="seconds", title="Cost of honoring one deletion")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="munl", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train a SISA ensemble and save it")
    _add_data_args(p)
    _add_sisa_args(p)
    p.add_argument("--out", default="runs/sisa.pt")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="report test metrics of a saved ensemble")
    p.add_argument("--model", default="runs/sisa.pt")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("unlearn", help="remove records (by Record Number) from a saved ensemble")
    p.add_argument("--model", default="runs/sisa.pt")
    p.add_argument("--ids", type=int, nargs="+", help="record numbers to forget")
    p.add_argument("--ids-file", help="file with whitespace-separated record numbers")
    p.add_argument("--out", help="where to write the updated ensemble (default: overwrite --model)")
    p.set_defaults(func=cmd_unlearn)

    p = sub.add_parser("benchmark", help="time SISA unlearning against full retraining")
    _add_data_args(p)
    _add_sisa_args(p)
    p.add_argument("--requests", type=int, default=10, help="sequential unlearning requests to serve")
    p.add_argument("--out", default="runs/benchmark")
    p.set_defaults(func=cmd_benchmark)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
