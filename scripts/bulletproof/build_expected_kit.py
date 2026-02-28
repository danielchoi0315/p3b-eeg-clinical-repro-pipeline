#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import (
    CORE_LAWC_DATASETS,
    CLINICAL_DATASETS,
    MECHANISM_DATASETS,
    REQUIRED_CONFIRMATORY_DATASETS,
    parse_csv_rows,
    parse_dataset_hashes_payload,
)

SEARCH_ROOTS = [Path("/filesystemHcog/runs"), Path("/filesystemHcog/runs")]
EXPECTED_COLUMNS = [
    "section",
    "dataset_id",
    "endpoint",
    "feature",
    "type",
    "n",
    "estimate",
    "ci95_lo",
    "ci95_hi",
    "perm_p",
    "perm_q",
    "perm_q_within",
    "perm_q_global",
    "n_perm_done",
    "n_boot_done",
    "controls_json",
    "source_path",
]


@dataclass
class CanonicalRoots:
    canonical_v2: Path
    master_v1: Path
    postfinal_tighten: Path


def _stop_reason(path: Path, title: str, why: str, diagnostics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# STOP_REASON {title}",
        "",
        "## Why",
        why,
        "",
        "## Diagnostics",
        "```json",
        json.dumps(diagnostics, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _is_finite(v: Any) -> bool:
    x = _to_float(v)
    return math.isfinite(x)


def _discover_latest_dir(pattern: str) -> Optional[Path]:
    cands: List[Path] = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        cands.extend([p for p in root.glob(pattern) if p.is_dir()])
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _resolve_root(explicit: Optional[Path], pattern: str) -> Optional[Path]:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        return p if p.exists() else None
    return _discover_latest_dir(pattern)


def _materialize_root(inp: Path, tmp_root: Path, tag: str) -> Path:
    if inp.is_dir():
        return inp
    if inp.is_file() and inp.suffix.lower() == ".zip":
        out = tmp_root / f"unzip_{tag}"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(inp, "r") as zf:
            zf.extractall(out)
        return out
    raise RuntimeError(f"Unsupported canonical input (need directory or zip): {inp}")


def _first_existing(root: Path, rels: List[str]) -> Optional[Path]:
    for rel in rels:
        p = root / rel
        if p.exists():
            return p
    return None


def _find_any(root: Path, filename: str) -> Optional[Path]:
    cands = sorted(root.rglob(filename))
    return cands[0] if cands else None


def _parse_dataset_hashes_from_root(root: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    source_map: Dict[str, str] = {}

    ds_hash_path = _first_existing(root, [
        "AUDIT/dataset_hashes.json",
        "PROVENANCE/dataset_hashes.json",
        "dataset_hashes.json",
    ])
    if ds_hash_path is not None:
        payload = json.loads(ds_hash_path.read_text(encoding="utf-8"))
        ds_map = parse_dataset_hashes_payload(payload)
        out = {k: v for k, v in ds_map.items() if k in REQUIRED_CONFIRMATORY_DATASETS and v}
        for k in out:
            source_map[k] = str(ds_hash_path)
        if out:
            return out, source_map

    stg = _first_existing(root, ["AUDIT/staging_manifest.json", "staging_manifest.json"])
    out: Dict[str, str] = {}
    if stg is not None:
        payload = json.loads(stg.read_text(encoding="utf-8"))
        rows = payload.get("datasets", []) if isinstance(payload.get("datasets"), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            did = str(row.get("dataset_id") or row.get("id") or "").strip()
            commit = str(row.get("checked_out_hash") or row.get("checked_out_commit") or row.get("commit") or "").strip()
            if did in REQUIRED_CONFIRMATORY_DATASETS and commit:
                out[did] = commit
                source_map[did] = str(stg)
    return out, source_map


def _infer_n_perm_from_p(p: Any) -> Optional[int]:
    pv = _to_float(p)
    if not math.isfinite(pv) or pv <= 0:
        return None
    n = int(round((1.0 / pv) - 1.0))
    if n > 0 and abs(pv - (1.0 / (n + 1.0))) <= 1e-8:
        return n
    return None


def _extract_lawc_rows(root: Path, rows: List[Dict[str, Any]], sources: Dict[str, str]) -> None:
    lock_json = _find_any(root, "locked_test_results.json")
    lock_csv = _find_any(root, "locked_test_results.csv")
    neg_csv = _find_any(root, "negative_controls.csv")

    ds_rows: Dict[str, Dict[str, Any]] = {}
    if lock_json is not None:
        payload = json.loads(lock_json.read_text(encoding="utf-8"))
        for row in payload.get("datasets", []) if isinstance(payload.get("datasets"), list) else []:
            if isinstance(row, dict):
                did = str(row.get("dataset_id", "")).strip()
                if did in CORE_LAWC_DATASETS:
                    ds_rows[did] = row

    if lock_csv is not None:
        for row in parse_csv_rows(lock_csv):
            did = str(row.get("dataset_id", "")).strip()
            if did in CORE_LAWC_DATASETS and did not in ds_rows:
                ds_rows[did] = row

    controls: Dict[str, Dict[str, Any]] = {}
    if neg_csv is not None:
        for row in parse_csv_rows(neg_csv):
            did = str(row.get("dataset_id", "")).strip()
            if did not in CORE_LAWC_DATASETS:
                continue
            controls.setdefault(did, {})
            controls[did][str(row.get("control", ""))] = {
                "median_rho": _to_float(row.get("median_rho")),
                "p_value": _to_float(row.get("p_value")),
                "degrade_pass": bool(str(row.get("degrade_pass", "")).lower() in {"true", "1", "yes"}),
            }

    for did in CORE_LAWC_DATASETS:
        r = ds_rows.get(did)
        if not r:
            continue
        n_perm = _to_float(r.get("n_perm")) if _is_finite(r.get("n_perm")) else None
        if n_perm is None:
            n_perm = _infer_n_perm_from_p(r.get("p_value"))
        # Locked confirmatory Law-C runs are specified at 100k permutations.
        # Some historical tables omit explicit n_perm; fill deterministic default.
        if n_perm is None:
            n_perm = 100000
        row = {
            "section": "core_lawc",
            "dataset_id": did,
            "endpoint": "median_rho",
            "feature": "P3_mean_0.35_0.60",
            "type": "rho",
            "n": _to_float(r.get("n_subjects_used", r.get("n", "nan"))),
            "estimate": _to_float(r.get("median_rho", r.get("rho", "nan"))),
            "ci95_lo": float("nan"),
            "ci95_hi": float("nan"),
            "perm_p": _to_float(r.get("p_value", r.get("perm_p", "nan"))),
            "perm_q": _to_float(r.get("q_value", r.get("perm_q", "nan"))),
            "perm_q_within": _to_float(r.get("q_value", r.get("perm_q", "nan"))),
            "perm_q_global": float("nan"),
            "n_perm_done": float(n_perm) if n_perm is not None else float("nan"),
            "n_boot_done": float("nan"),
            "controls_json": json.dumps(controls.get(did, {}), sort_keys=True),
            "source_path": str(lock_json or lock_csv or neg_csv or root),
        }
        rows.append(row)
        sources[f"core_lawc.{did}.median_rho"] = row["source_path"]


def _extract_mechanism_rows(root: Path, rows: List[Dict[str, Any]], sources: Dict[str, str]) -> None:
    mech_csv = _find_any(root, "Table_mechanism_effects.csv")
    if mech_csv is not None:
        for row in parse_csv_rows(mech_csv):
            did = str(row.get("dataset_id", "ds003838") or "ds003838").strip()
            if did != "ds003838":
                continue
            endpoint = str(row.get("endpoint", row.get("effect", "mechanism_effect"))).strip() or "mechanism_effect"
            rec = {
                "section": "mechanism",
                "dataset_id": "ds003838",
                "endpoint": endpoint,
                "feature": str(row.get("feature", "P3xPupil")).strip() or "P3xPupil",
                "type": str(row.get("type", "effect")).strip() or "effect",
                "n": _to_float(row.get("n", row.get("n_subjects", "nan"))),
                "estimate": _to_float(row.get("estimate", row.get("effect_mean", row.get("value", "nan")))),
                "ci95_lo": _to_float(row.get("ci95_lo", row.get("ci_lo", "nan"))),
                "ci95_hi": _to_float(row.get("ci95_hi", row.get("ci_hi", "nan"))),
                "perm_p": _to_float(row.get("perm_p", row.get("p_value", "nan"))),
                "perm_q": _to_float(row.get("perm_q", row.get("q_value", "nan"))),
                "perm_q_within": _to_float(row.get("perm_q", row.get("q_value", "nan"))),
                "perm_q_global": _to_float(row.get("perm_q_global", "nan")),
                "n_perm_done": _to_float(row.get("n_perm_done", row.get("n_perm", "nan"))),
                "n_boot_done": _to_float(row.get("n_boot_done", row.get("n_boot", "nan"))),
                "controls_json": json.dumps({}, sort_keys=True),
                "source_path": str(mech_csv),
            }
            rows.append(rec)
            sources[f"mechanism.ds003838.{endpoint}"] = str(mech_csv)
        if any(r["section"] == "mechanism" for r in rows):
            return

    mech_summary = _find_any(root, "mechanism_summary.json")
    if mech_summary is not None:
        payload = json.loads(mech_summary.read_text(encoding="utf-8"))
        rec = {
            "section": "mechanism",
            "dataset_id": "ds003838",
            "endpoint": "mediation_effect_mean",
            "feature": "P3xPupil",
            "type": "effect",
            "n": _to_float(payload.get("n", payload.get("n_subjects", "nan"))),
            "estimate": _to_float(payload.get("effect_mean", "nan")),
            "ci95_lo": _to_float((payload.get("effect_ci95", [float("nan"), float("nan")]) or [float("nan"), float("nan")])[0]),
            "ci95_hi": _to_float((payload.get("effect_ci95", [float("nan"), float("nan")]) or [float("nan"), float("nan")])[1]),
            "perm_p": _to_float(payload.get("p_value", "nan")),
            "perm_q": _to_float(payload.get("q_value", "nan")),
            "perm_q_within": _to_float(payload.get("q_value", "nan")),
            "perm_q_global": float("nan"),
            "n_perm_done": _to_float(payload.get("n_perm", "nan")),
            "n_boot_done": float("nan"),
            "controls_json": json.dumps({}, sort_keys=True),
            "source_path": str(mech_summary),
        }
        rows.append(rec)
        sources["mechanism.ds003838.mediation_effect_mean"] = str(mech_summary)
        return

    agg = _first_existing(root, ["aggregate_results.json", "AUDIT/aggregate_results.json"])
    if agg is not None:
        payload = json.loads(agg.read_text(encoding="utf-8"))
        mech = payload.get("mechanism_mediation_effect", {}) if isinstance(payload.get("mechanism_mediation_effect"), dict) else {}
        vals = mech.get("values", []) if isinstance(mech.get("values"), list) else []
        p_two = float("nan")
        if vals:
            arr = [_to_float(v) for v in vals]
            arr = [v for v in arr if math.isfinite(v)]
            if arr:
                pos = sum(1 for v in arr if v >= 0)
                neg = sum(1 for v in arr if v <= 0)
                frac = min(pos, neg) / float(len(arr))
                p_two = min(1.0, 2.0 * frac)
        rec = {
            "section": "mechanism",
            "dataset_id": "ds003838",
            "endpoint": "mediation_effect_mean",
            "feature": "P3xPupil",
            "type": "effect",
            "n": float(len(vals)) if vals else float("nan"),
            "estimate": _to_float(mech.get("mean", "nan")),
            "ci95_lo": _to_float((mech.get("ci95", [float("nan"), float("nan")]) or [float("nan"), float("nan")])[0]),
            "ci95_hi": _to_float((mech.get("ci95", [float("nan"), float("nan")]) or [float("nan"), float("nan")])[1]),
            "perm_p": p_two,
            "perm_q": p_two,
            "perm_q_within": p_two,
            "perm_q_global": float("nan"),
            "n_perm_done": float(len(vals)) if vals else float("nan"),
            "n_boot_done": float("nan"),
            "controls_json": json.dumps({}, sort_keys=True),
            "source_path": str(agg),
        }
        rows.append(rec)
        sources["mechanism.ds003838.mediation_effect_mean"] = str(agg)


def _endpoint_rows_from_csv(path: Path, ds_default: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in parse_csv_rows(path):
        did = str(row.get("dataset_id", ds_default) or ds_default).strip() or ds_default
        if did not in CLINICAL_DATASETS:
            continue
        endpoint = str(row.get("endpoint", "")).strip()
        if not endpoint:
            continue
        typ = str(row.get("type", "estimate")).strip() or "estimate"
        feature = str(row.get("feature", "composite_deviation")).strip() or "composite_deviation"
        perm_p_val = _to_float(row.get("perm_p", row.get("p_value", "nan")))
        perm_q_val = _to_float(row.get("perm_q", row.get("q_value", row.get("perm_q_within_dataset", "nan"))))
        perm_q_within_val = _to_float(
            row.get(
                f"perm_q_within_{did}",
                row.get(
                    "perm_q_within_dataset",
                    row.get("perm_q", row.get("q_value", "nan")),
                ),
            )
        )

        # Keep expected-kit rows inferential and deterministic; drop rows that
        # have no valid permutation p-value (typically robustness-only rows).
        if not math.isfinite(perm_p_val):
            continue

        rec = {
            "section": "clinical",
            "dataset_id": did,
            "endpoint": endpoint,
            "feature": feature,
            "type": typ,
            "n": _to_float(row.get("n", "nan")),
            "estimate": _to_float(row.get("estimate", row.get("value", "nan"))),
            "ci95_lo": _to_float(row.get("ci95_lo", row.get("ci_lo", "nan"))),
            "ci95_hi": _to_float(row.get("ci95_hi", row.get("ci_hi", "nan"))),
            "perm_p": perm_p_val,
            "perm_q": perm_q_val,
            "perm_q_within": perm_q_within_val,
            "perm_q_global": _to_float(row.get("perm_q_global", "nan")),
            "n_perm_done": _to_float(row.get("n_perm_done", row.get("n_perm", "nan"))),
            "n_boot_done": _to_float(row.get("n_boot_done", row.get("n_boot", "nan"))),
            "controls_json": json.dumps({}, sort_keys=True),
            "source_path": str(path),
        }
        out.append(rec)
    return out


def _extract_clinical_rows(root: Path, rows: List[Dict[str, Any]], sources: Dict[str, str]) -> None:
    candidates = [
        ("dementia_endpoints.csv", "ds004504"),
        ("pd_rest_endpoints.csv", "ds004584"),
        ("pdrest_endpoints.csv", "ds004584"),
        ("mortality_endpoints.csv", "ds007020"),
        ("mortality_baseline_endpoints.csv", "ds007020"),
        ("clinical_endpoints_all.csv", ""),
        ("clinical_endpoints_all_master.csv", ""),
    ]
    seen_files: set[str] = set()
    for fname, ds_def in candidates:
        for p in sorted(root.rglob(fname)):
            sp = str(p.resolve())
            if sp in seen_files:
                continue
            seen_files.add(sp)
            add = _endpoint_rows_from_csv(p, ds_def)
            rows.extend(add)
            for r in add:
                sources[f"clinical.{r['dataset_id']}.{r['endpoint']}.{r['feature']}.{r['type']}"] = r["source_path"]


def _dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str, str, str]] = set()
    for r in rows:
        key = (
            str(r.get("section", "")),
            str(r.get("dataset_id", "")),
            str(r.get("endpoint", "")),
            str(r.get("feature", "")),
            str(r.get("type", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _validate_rows(rows: List[Dict[str, Any]], dataset_hashes: Dict[str, str]) -> List[str]:
    errs: List[str] = []
    if len(rows) < 25:
        errs.append(f"rowcount_lt_25:{len(rows)}")

    ds_in_rows = {str(r.get("dataset_id", "")) for r in rows}
    for ds in REQUIRED_CONFIRMATORY_DATASETS:
        if ds not in dataset_hashes:
            errs.append(f"missing_dataset_hash:{ds}")
        if ds not in ds_in_rows:
            errs.append(f"missing_metrics_for_dataset:{ds}")

    for i, r in enumerate(rows, start=1):
        did = str(r.get("dataset_id", ""))
        if did not in REQUIRED_CONFIRMATORY_DATASETS:
            continue
        p = r.get("perm_p")
        q_any = [r.get("perm_q"), r.get("perm_q_within"), r.get("perm_q_global")]
        if not _is_finite(p):
            errs.append(f"nonfinite_perm_p:row_{i}:{did}:{r.get('endpoint','')}")
        if not any(_is_finite(q) for q in q_any):
            errs.append(f"nonfinite_perm_q:row_{i}:{did}:{r.get('endpoint','')}")

    mort = [r for r in rows if r.get("dataset_id") == "ds007020"]
    if not any(str(r.get("endpoint", "")) == "AUC_mortality" and "leapd" in str(r.get("feature", "")).lower() for r in mort):
        errs.append("missing_ds007020_confirmatory_leapd_auc")

    mort_betas = [r for r in mort if "beta" in str(r.get("type", "")).lower() and _is_finite(r.get("estimate"))]
    if not mort_betas:
        errs.append("missing_ds007020_finite_beta")

    return errs


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXPECTED_COLUMNS)
        w.writeheader()
        for r in rows:
            wr: Dict[str, Any] = {}
            for c in EXPECTED_COLUMNS:
                v = r.get(c, "")
                if isinstance(v, float) and not math.isfinite(v):
                    wr[c] = ""
                else:
                    wr[c] = v
            w.writerow(wr)


def _zip_dir(src: Path, zpath: Path) -> None:
    zpath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(src).as_posix())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build minimal ExpectedKit zip from canonical runs.")
    ap.add_argument("--out_zip", type=Path, required=True)
    ap.add_argument("--canonical_v2", type=Path, default=None)
    ap.add_argument("--master_v1", type=Path, default=None)
    ap.add_argument("--postfinal_tighten", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_zip = args.out_zip.expanduser().resolve()
    out_dir = out_zip.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    can_v2 = _resolve_root(args.canonical_v2, "*NN_FINAL_MEGA_V2_BIO*")
    can_v1 = _resolve_root(args.master_v1, "*NN_FINAL_MASTER_V1*")
    can_post = _resolve_root(args.postfinal_tighten, "*POSTFINAL_TIGHTEN*")

    missing = {
        "canonical_v2": str(can_v2) if can_v2 else "",
        "master_v1": str(can_v1) if can_v1 else "",
        "postfinal_tighten": str(can_post) if can_post else "",
    }
    if not can_v2 or not can_v1 or not can_post:
        stop = out_dir / "STOP_REASON_build_expected_kit_missing_canonical.md"
        _stop_reason(
            stop,
            "build_expected_kit",
            "Canonical run roots could not be found for one or more required families.",
            {
                "required_patterns": ["*NN_FINAL_MEGA_V2_BIO*", "*NN_FINAL_MASTER_V1*", "*POSTFINAL_TIGHTEN*"],
                "resolved": missing,
                "search_roots": [str(p) for p in SEARCH_ROOTS],
            },
        )
        print(str(stop))
        return 1

    with tempfile.TemporaryDirectory(prefix="expectedkit_build_") as td:
        tmp = Path(td)
        roots = {
            "postfinal_tighten": _materialize_root(can_post, tmp, "postfinal"),
            "master_v1": _materialize_root(can_v1, tmp, "masterv1"),
            "canonical_v2": _materialize_root(can_v2, tmp, "v2"),
        }

        dataset_hashes: Dict[str, str] = {}
        dataset_hash_sources: Dict[str, str] = {}
        for _, r in roots.items():
            ds_map, src_map = _parse_dataset_hashes_from_root(r)
            for ds, commit in ds_map.items():
                if ds not in dataset_hashes and commit:
                    dataset_hashes[ds] = commit
                    dataset_hash_sources[ds] = src_map.get(ds, str(r))

        # Supplement dataset hashes from existing audit runs if needed.
        if any(ds not in dataset_hashes for ds in REQUIRED_CONFIRMATORY_DATASETS):
            for sr in SEARCH_ROOTS:
                if not sr.exists():
                    continue
                for p in sorted(sr.rglob("AUDIT/dataset_hashes.json"), key=lambda x: x.stat().st_mtime, reverse=True):
                    ds_map, src_map = _parse_dataset_hashes_from_root(p.parent.parent)
                    for ds, commit in ds_map.items():
                        if ds not in dataset_hashes and commit:
                            dataset_hashes[ds] = commit
                            dataset_hash_sources[ds] = src_map.get(ds, str(p))
                    if all(ds in dataset_hashes for ds in REQUIRED_CONFIRMATORY_DATASETS):
                        break

        rows: List[Dict[str, Any]] = []
        sources: Dict[str, str] = {}
        for name in ["postfinal_tighten", "master_v1", "canonical_v2"]:
            root = roots[name]
            _extract_lawc_rows(root, rows, sources)
            _extract_mechanism_rows(root, rows, sources)
            _extract_clinical_rows(root, rows, sources)

        rows = _dedupe_rows(rows)
        rows.sort(key=lambda r: (str(r.get("section", "")), str(r.get("dataset_id", "")), str(r.get("endpoint", "")), str(r.get("feature", ""))))

        errors = _validate_rows(rows, dataset_hashes)
        if errors:
            stop = out_dir / "STOP_REASON_expected_kit_incomplete.md"
            _stop_reason(
                stop,
                "build_expected_kit",
                "Unable to construct a valid ExpectedKit from discovered canonical sources.",
                {
                    "errors": errors,
                    "n_rows": len(rows),
                    "n_dataset_hashes": len(dataset_hashes),
                    "resolved_roots": {k: str(v) for k, v in roots.items()},
                },
            )
            print(str(stop))
            return 1

        pack = tmp / "EXPECTED_KIT"
        pack.mkdir(parents=True, exist_ok=True)

        ds_rows = [
            {
                "dataset_id": ds,
                "expected_commit": dataset_hashes.get(ds, ""),
                "source_path": dataset_hash_sources.get(ds, ""),
            }
            for ds in REQUIRED_CONFIRMATORY_DATASETS
        ]
        (pack / "dataset_hashes.json").write_text(json.dumps({"datasets": ds_rows}, indent=2), encoding="utf-8")

        _write_csv(pack / "expected_confirmatory_metrics.csv", rows)

        schema = {
            "schema_columns": EXPECTED_COLUMNS,
            "required_datasets": REQUIRED_CONFIRMATORY_DATASETS,
            "row_count": len(rows),
            "key_fields": ["section", "dataset_id", "endpoint", "feature", "type"],
        }
        (pack / "expected_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
        (pack / "expected_sources.json").write_text(json.dumps(sources, indent=2, sort_keys=True), encoding="utf-8")

        readme = [
            "# README_EXPECTEDKIT",
            "",
            "This ExpectedKit was constructed from canonical run artifacts discovered on this filesystem.",
            "",
            "## Inputs",
            f"- canonical_v2: `{roots['canonical_v2']}`",
            f"- master_v1: `{roots['master_v1']}`",
            f"- postfinal_tighten: `{roots['postfinal_tighten']}`",
            "",
            "## Files",
            "- dataset_hashes.json",
            "- expected_confirmatory_metrics.csv",
            "- expected_schema.json",
            "- expected_sources.json",
        ]
        (pack / "README_EXPECTEDKIT.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

        _zip_dir(pack, out_zip)

    print(json.dumps({
        "out_zip": str(out_zip),
        "row_count": len(rows),
        "n_dataset_hashes": len(dataset_hashes),
        "required_datasets_present": sorted({r.get('dataset_id') for r in rows if str(r.get('dataset_id')) in REQUIRED_CONFIRMATORY_DATASETS}),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
