"""Offline domain-characterization driver (Component C1, Phase 1).

Streams each configured dataset (full stream: train + test) through the
backbone, computes the ``DomainDescriptor`` (centroid + image statistics +
``classes_seen``) and emits a fully readable Phase 1 acceptance report:

- ``outputs/characterize/descriptors.json`` — descriptors by (model, dataset)
- ``outputs/characterize/data_provenance.json`` — real per-split counts & label
  sets (the "measure, don't assume" pass; decides how APTOS is composed)
- ``outputs/characterize/phase1_acceptance_{model}.json`` / ``.md`` — verdict
- ``outputs/characterize/similarity_matrix_{model}.csv`` + printed ASCII table
- ``outputs/characterize/phase1_report.md`` — one combined human-readable report

Design notes (see Agentic_CTTA_New_Architecture.md §5 and the Phase-1 plan):

- Streams are deterministic: ``augment=False``, ``shuffle=False``, fixed seed.
- ``classes_seen`` is label-derived **only over the labeled subset of the
  stream** (train split when the test split carries placeholder labels — e.g.
  an unlabeled competition test set). Never fabricated.
- The acceptance gate now checks the *absolute* reachability of the bank's
  match threshold (share of within-half pairs >= match_threshold), not just
  separation — fixing the "separation passes but 0.95 is unreachable" trap.
  The recommended ``match_threshold`` is reported from the measured within
  distribution (data-driven, default recommendation p10 of within cosines).

Run (local smoke is impossible without data/model weights; run on Kaggle):

    python scripts/run_characterize.py --model both \
        --datasets idrid,aptos2019,messidor2 \
        --data-dir idrid=/kaggle/input/idrid \
        --out outputs/characterize
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.config import CTTAConfig, ModelConfig
from src.data.registry import DatasetRegistry
from src.data.fullstream import (
    build_raw_stream,
    split_row_counts,
    stream_provenance,
)
from src.models.registry import ModelRegistry
from src.agents.characterize import Characterizer, write_descriptors_json
from src.agents.evaluation import (
    DEFAULT_MATCH_THRESHOLD,
    retrieval_accuracy,
    similarity_matrix,
    within_between_analysis,
)

logger = logging.getLogger(__name__)
DEFAULT_DATA_DIRS = {
    "idrid": "./data/IDRiD/",
    "aptos2019": "./data/APTOS2019/",
    "messidor2": "./data/MESSIDOR2/",
}

DEFAULT_IMAGE_SIZE = 224


class _DatasetSpec:
    def __init__(self, name: str, data_dir: str, image_size: int):
        self.name = name
        self.data_dir = data_dir
        self.image_size = image_size


def _parse_data_dirs(overrides: List[str]) -> Dict[str, str]:
    data_dirs = dict(DEFAULT_DATA_DIRS)
    for ov in overrides or []:
        if "=" not in ov:
            raise SystemExit(
                f"--data-dir expects NAME=PATH, got {ov!r} (e.g. --data-dir idrid=/kaggle/input/idrid)"
            )
        name, path = ov.split("=", 1)
        data_dirs[name.strip()] = path.strip()
    return data_dirs


def _fmt_table(header: List[str], rows: List[List[Any]], title: str = "") -> str:
    """Fixed-width ASCII table for copy-paste / notebook display."""
    col_w = [len(str(h)) for h in header]
    for r in rows:
        for i, v in enumerate(r):
            col_w[i] = max(col_w[i], len(_num(v)))
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    def line(vals):
        return "| " + " | ".join(_num(v).ljust(col_w[i]) for i, v in enumerate(vals)) + " |"
    out = [sep]
    if title:
        out.append(f"# {title}")
    out.append(line(header))
    out.append(sep)
    for r in rows:
        out.append(line(r))
    out.append(sep)
    return "\n".join(out)


def _num(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _data_config_from_arg(name: str, data_dir: str) -> _DatasetSpec:
    return _DatasetSpec(name=name, data_dir=data_dir, image_size=DEFAULT_IMAGE_SIZE)


def _label_counts_to_array(label_counts: Dict[int, int], num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.float64)
    for k, v in label_counts.items():
        if 0 <= int(k) < num_classes:
            counts[int(k)] = float(v)
    return counts


def _characterize_domain(
    characterizer: Characterizer,
    full_stream,
    class_counts: np.ndarray | None,
    spec: _DatasetSpec,
    partition_counts: Dict[str, int],
    labeled_n: int,
    batch_size: int,
    device: torch.device,
    label: str = "full",
):
    full_loader = (
        full_stream if isinstance(full_stream, DataLoader)
        else DataLoader(full_stream, batch_size=batch_size, shuffle=False)
    )
    t0 = time.perf_counter()
    desc = characterizer.characterize(
        full_loader,
        domain_name=spec.name,
        partition_counts=partition_counts,
        class_counts=class_counts,
        pass_label=label,
    )
    dt = time.perf_counter() - t0
    return desc, dt


def run_for_model(
    model_name: str,
    specs: List[_DatasetSpec],
    out_dir: Path,
    match_threshold: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    with open(f"configs/model/{model_name}.yaml", "r", encoding="utf-8") as f:
        mcfg = ModelConfig(**yaml.safe_load(f))
    model_class = ModelRegistry.get(mcfg.name)
    model = model_class(
        num_classes=mcfg.num_classes,
        freeze_layers=mcfg.freeze_layers,
        checkpoint=mcfg.checkpoint,
        image_size=mcfg.image_size,
        drop_path=mcfg.drop_path,
        normalize_features=mcfg.normalize_features,
    )
    model.load_weights()
    model = model.to(device)
    use_labels = CTTAConfig().use_labels_for_classes_seen

    descriptors: Dict[str, Any] = {}
    reps_by_domain: Dict[str, List[Any]] = {}
    timing: Dict[str, float] = {}

    for spec in specs:
        provenance = stream_provenance(spec.name, spec.data_dir, spec.image_size)
        logger.info(
            f"[{model_name}] {spec.name}: stream={provenance['n_stream']} "
            f"(train={provenance['n_train']}, test={provenance['n_test']}) "
            f"suspicious_placeholder_test={provenance['suspicious_placeholder_test']}"
        )
        full_stream = build_raw_stream(spec)
        split_counts = provenance["splits"]
        if provenance["suspicious_placeholder_test"]:
            label_counts = split_counts["train"]["label_counts"]
            labeled_n = provenance["n_train"]
            logger.info(
                f"[{model_name}] {spec.name}: test labels look like placeholders; "
                f"classes_seen from train split only ({labeled_n} samples)"
            )
        else:
            label_counts = {
                k: split_counts["train"]["label_counts"].get(k, 0)
                + split_counts["test"]["label_counts"].get(k, 0)
                for k in set(split_counts["train"]["label_counts"])
                | set(split_counts["test"]["label_counts"])
            }
            labeled_n = provenance["n_stream"]
        class_counts = _label_counts_to_array(label_counts, mcfg.num_classes)
        partition_counts = {"train": provenance["n_train"], "test": provenance["n_test"]}

        characterizer = Characterizer(
            model=model,
            model_name=model_name,
            num_classes=mcfg.num_classes,
            normalize_mean=mcfg.normalize_mean,
            normalize_std=mcfg.normalize_std,
            device=device,
            use_labels_for_classes_seen=use_labels,
            seed=seed,
        )
        desc, dt = _characterize_domain(
            characterizer, full_stream, class_counts, spec,
            partition_counts, labeled_n, batch_size, device, label="full",
        )
        timing[spec.name] = dt
        descriptors[spec.name] = desc
        logger.info(f"[{model_name}] {spec.name}: full-stream pass done in {dt:.1f}s")

        n = len(full_stream)
        half_a_ds = Subset(full_stream, list(range(0, n, 2)))
        half_b_ds = Subset(full_stream, list(range(1, n, 2)))
        # split-half descriptors re-characterize each half's stream
        desc_a, dt_a = _characterize_domain(
            characterizer, half_a_ds, class_counts, spec, partition_counts,
            labeled_n, batch_size, device, label="half-A")
        desc_b, dt_b = _characterize_domain(
            characterizer, half_b_ds, class_counts, spec, partition_counts,
            labeled_n, batch_size, device, label="half-B")
        logger.info(
            f"[{model_name}] {spec.name}: half-A {dt_a:.1f}s, half-B {dt_b:.1f}s")
        reps_by_domain[spec.name] = [desc_a, desc_b]

    report = within_between_analysis(reps_by_domain, match_threshold=match_threshold)
    retrieval = retrieval_accuracy(
        reference=[reps_by_domain[d][0] for d in reps_by_domain],
        queries=[reps_by_domain[d][1] for d in reps_by_domain],
    )
    sim = similarity_matrix(descriptors)
    names = list(descriptors)

    # Data-driven recommendation: floor of min_within to 2dp (guaranteed
    # reachable: every within pair >= min_within >= recommended), floored at 0.80.
    if report["n_within_pairs"]:
        reachable = min(report["min_within"], report["p10_within"])
        recommended = max(0.80, math.floor(reachable * 100) / 100)
    else:
        recommended = None

    return {
        "model": model_name,
        "descriptors": descriptors,
        "reps": reps_by_domain,
        "acceptance": {**report, "recommended_match_threshold": recommended,
                       "match_threshold_used": match_threshold},
        "retrieval": retrieval,
        "similarity": {"names": names, "matrix": sim.tolist()},
        "timing_seconds": timing,
    }


def _write_model_artifacts(result: Dict[str, Any], out_dir: Path) -> Path:
    model = result["model"]
    out_dir.mkdir(parents=True, exist_ok=True)
    descriptors = {d.domain_name: d for d in result["descriptors"].values()}
    write_descriptors_json(descriptors, out_dir / f"descriptors_{model}.json")

    acc = result["acceptance"]
    acc_json = {
        **{k: acc[k] for k in ("within_mean", "between_mean", "min_within",
                               "max_between", "p10_within", "p50_within",
                               "separation", "match_threshold_used",
                               "recommended_match_threshold",
                               "share_within_ge_threshold", "n_within_pairs",
                               "pass_sep", "pass_abs", "pass")},
        "retrieval_top1": result["retrieval"]["top1_accuracy"],
        "timing_seconds": result["timing_seconds"],
    }
    (out_dir / f"phase1_acceptance_{model}.json").write_text(
        json.dumps(acc_json, indent=2, sort_keys=True))

    names = result["similarity"]["names"]
    mat = np.asarray(result["similarity"]["matrix"])
    csv_lines = ["domain," + ",".join(names)]
    for i, n in enumerate(names):
        csv_lines.append(f"{n}," + ",".join(f"{mat[i, j]:.4f}" for j in range(len(names))))
    (out_dir / f"similarity_matrix_{model}.csv").write_text("\n".join(csv_lines))

    acc_rows = [
        ["within_mean", f"{acc['within_mean']:.4f}"],
        ["between_mean", f"{acc['between_mean']:.4f}"],
        ["min_within", f"{acc['min_within']:.4f}"],
        ["max_between", f"{acc['max_between']:.4f}"],
        ["p10_within", f"{acc['p10_within']:.4f}"],
        ["p50_within", f"{acc['p50_within']:.4f}"],
        ["separation", f"{acc['separation']:.4f}"],
        [f"share_within >= {acc['match_threshold_used']}",
         f"{acc['share_within_ge_threshold']:.3f}"],
        ["n_within_pairs", str(acc["n_within_pairs"])],
        ["retrieval_top1", f"{result['retrieval']['top1_accuracy']:.3f}"],
        ["recommended_match_threshold", f"{acc['recommended_match_threshold']}"],
        ["PASS", str(acc["pass"])],
    ]
    md = [f"# Phase 1 acceptance — {model}", "",
          _fmt_table(["metric", "value"], acc_rows), "",
          "## Similarity matrix (centroid cosine)", "",
          _fmt_table(["domain"] + names,
                     [[n] + [f"{mat[i, j]:.4f}" for j in range(len(names))]
                      for i, n in enumerate(names)]), "",
          "## Timing (seconds)", "",
          _fmt_table(["dataset", "seconds"],
                     [[n, f"{t:.2f}"] for n, t in result["timing_seconds"].items()]),
          ]
    (out_dir / f"phase1_acceptance_{model}.md").write_text("\n".join(md))
    return out_dir / f"phase1_acceptance_{model}.md"


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="C1 domain characterization (Phase 1)")
    parser.add_argument("--model", default="both",
                        choices=["retfound", "visionfm", "both"])
    parser.add_argument("--datasets", default="idrid,aptos2019,messidor2")
    parser.add_argument("--data-dir", action="append", default=[],
                        help="NAME=PATH override (repeatable)")
    parser.add_argument("--match-threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/characterize")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dirs = _parse_data_dirs(args.data_dir)
    specs = [_data_config_from_arg(n, data_dirs[n]) for n in args.datasets.split(",")]
    if len(specs) < 2:
        raise SystemExit(
            "Phase 1 acceptance needs >= 2 domains (within/between comparison). "
            "Pass --datasets idrid,aptos2019,messidor2")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    provenances = {s.name: stream_provenance(s.name, s.data_dir, s.image_size)
                   for s in specs}
    (out_dir / "data_provenance.json").write_text(
        json.dumps(provenances, indent=2, sort_keys=True))
    print("\n================ DATA PROVENANCE ================")
    for s in specs:
        p = provenances[s.name]
        print(f"{s.name}: train={p['n_train']} test={p['n_test']} "
              f"stream={p['n_stream']} unique_labels={p['unique_labels_per_split']} "
              f"placeholder_test={p['suspicious_placeholder_test']}")
    print("==================================================\n")

    models = ["retfound", "visionfm"] if args.model == "both" else [args.model]
    md_sections = []
    for m in models:
        result = run_for_model(m, specs, out_dir, args.match_threshold,
                               args.batch_size, args.seed, device)
        md_path = _write_model_artifacts(result, out_dir)
        acc = result["acceptance"]
        print(f"\n=============== Phase 1 acceptance — {m} ===============")
        print(f"within_mean={acc['within_mean']:.4f} between_mean={acc['between_mean']:.4f} "
              f"min_within={acc['min_within']:.4f} max_between={acc['max_between']:.4f} "
              f"p50_within={acc['p50_within']:.4f} separation={acc['separation']:.4f}")
        print(f"retrieval_top1={result['retrieval']['top1_accuracy']:.3f} "
              f"recommended_match_threshold={acc['recommended_match_threshold']} "
              f"PASS={acc['pass']}")
        names = result["similarity"]["names"]
        mat = np.asarray(result["similarity"]["matrix"])
        print(_fmt_table(["domain"] + names,
                         [[n] + [f"{mat[i, j]:.4f}" for j in range(len(names))]
                          for i, n in enumerate(names)], title=f"{m} centroid similarity"))
        print(f"acceptance md: {md_path}")
        md_sections.append(md_path.read_text())

    combined = "\n\n---\n\n".join(md_sections)
    (out_dir / "phase1_report.md").write_text(combined)
    print(f"\nCombined report: {out_dir / 'phase1_report.md'}")


if __name__ == "__main__":
    main()
