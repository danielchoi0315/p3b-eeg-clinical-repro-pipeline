#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import detect_slurm, ensure_out_tree, ensure_stage_status, out_root_default, run_cmd, stop_reason, write_json


@dataclass
class StageDef:
    name: str
    script: str
    args: List[str]


def parse_args() -> argparse.Namespace:
    default_repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path, default=out_root_default())
    ap.add_argument("--repo_root", type=Path, default=Path(os.environ.get("REPO_ROOT", str(default_repo))))
    ap.add_argument("--data_root", type=Path, default=Path("/lambda/nfs/HCog/filesystemHcog/openneuro"))
    ap.add_argument("--wall_hours", type=float, default=12.0)
    ap.add_argument("--expected_kit", type=Path, default=None)
    ap.add_argument("--max_attempts", type=int, default=10)
    return ap.parse_args()


def _tail(path: Path, n: int = 200) -> str:
    if not path.exists():
        return ""
    txt = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(txt[-n:])


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _auto_fix(stage: str, stage_log: Path, master_log: Path, out_root: Path, repo_root: Path) -> bool:
    tail = _tail(stage_log, n=240)
    low = tail.lower()
    fixed = False

    # Minimal fix 1: dependency/module issues.
    if "modulenotfounderror" in low or "no module named" in low:
        cmd = "python3 -m pip install --upgrade pip setuptools wheel && python3 -m pip install -r requirements.txt"
        subprocess.run(
            ["bash", "-lc", cmd],
            cwd=str(repo_root),
            check=False,
            stdout=master_log.open("a"),
            stderr=subprocess.STDOUT,
        )
        fixed = True

    # Minimal fix 2: common file naming mismatch metric.csv -> metrics.csv.
    if "metrics.csv" in low and "no such file" in low:
        cmd = (
            f"find {out_root / 'REPRO_FROM_SCRATCH'} -type f -name 'metric.csv' "
            "-print0 2>/dev/null | xargs -0 -I{} bash -lc 'mv \"{}\" \"${{}%/metric.csv}/metrics.csv\"'"
        )
        subprocess.run(["bash", "-lc", cmd], check=False, stdout=master_log.open("a"), stderr=subprocess.STDOUT)
        fixed = True

    # Minimal fix 3: missing keys in results payloads used by Stage3.
    if "keyerror" in low and ("n_perm_done" in low or "n_boot_done" in low):
        fixer = (
            "python3 - <<'PY'\n"
            "import json, pathlib\n"
            f"root = pathlib.Path({json.dumps(str(out_root / 'REPRO_FROM_SCRATCH'))})\n"
            "for p in root.rglob('results.json'):\n"
            "    try:\n"
            "        d=json.loads(p.read_text())\n"
            "    except Exception:\n"
            "        continue\n"
            "    changed=False\n"
            "    if 'n_perm_done' not in d:\n"
            "        d['n_perm_done']=0; changed=True\n"
            "    if 'n_boot_done' not in d:\n"
            "        d['n_boot_done']=0; changed=True\n"
            "    if changed:\n"
            "        p.write_text(json.dumps(d, indent=2))\n"
            "PY"
        )
        subprocess.run(["bash", "-lc", fixer], check=False, stdout=master_log.open("a"), stderr=subprocess.STDOUT)
        fixed = True

    # Minimal fix 4: Slurm submission issue -> force local Stage2 path.
    if stage == "stage2_repro_confirmatory" and "sbatch" in low and ("error" in low or "failed" in low):
        (out_root / "AUDIT" / "FORCE_STAGE2_LOCAL").write_text("1\n", encoding="utf-8")
        fixed = True

    if fixed:
        _append_log(master_log, f"[{time.time()}] auto_fix_applied stage={stage}")
    return fixed


def _run_stage(
    stage: StageDef,
    repo_root: Path,
    out_root: Path,
    extra_args: Dict[str, str],
    audit: Path,
    master_log: Path,
    max_attempts: int,
) -> bool:
    stage_log = audit / f"{stage.name}.log"
    cmd = ["python3", str(repo_root / stage.script), "--out_root", str(out_root)]
    for k, v in extra_args.items():
        if v is None or v == "":
            continue
        cmd.extend([f"--{k}", str(v)])
    cmd.extend(stage.args)

    for attempt in range(1, max_attempts + 1):
        _append_log(master_log, f"[{time.time()}] stage={stage.name} attempt={attempt} cmd={' '.join(cmd)}")
        rc = run_cmd(cmd, log_path=stage_log, allow_fail=True).rc
        if rc == 0:
            ensure_stage_status(audit, stage.name, "PASS", {"attempt": attempt, "log": str(stage_log)})
            return True

        tail = _tail(stage_log, 200)
        if stage.name in {"stage2_repro_confirmatory", "stage3_match_check"}:
            print(f"===== {stage.name} FAIL attempt={attempt} last_200_lines =====")
            print(tail)
            print(f"===== END {stage.name} last_200_lines =====")
            _append_log(master_log, f"[{time.time()}] {stage.name} tail:\n{tail}\n")

        fixed = _auto_fix(stage.name, stage_log, master_log, out_root, repo_root)
        if not fixed and attempt >= max_attempts:
            ensure_stage_status(audit, stage.name, "FAIL", {"attempt": attempt, "log": str(stage_log), "tail": tail})
            return False
        if not fixed and stage.name not in {"stage2_repro_confirmatory", "stage3_match_check"}:
            ensure_stage_status(audit, stage.name, "FAIL", {"attempt": attempt, "log": str(stage_log), "tail": tail})
            return False

    ensure_stage_status(audit, stage.name, "FAIL", {"attempt": max_attempts, "log": str(stage_log)})
    return False


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_root = args.out_root
    paths = ensure_out_tree(out_root)
    audit = paths["AUDIT"]
    master_log = audit / "master_monitor.log"

    slurm = detect_slurm()
    write_json(audit / "slurm_detection.json", slurm)

    stages = [
        StageDef("stage0_preflight", "scripts/bulletproof/stage0_preflight.py", []),
        StageDef("stage1_stage_raw", "scripts/bulletproof/stage1_stage_raw.py", []),
        StageDef("stage2_repro_confirmatory", "scripts/bulletproof/stage2_repro_confirmatory.py", []),
        StageDef("stage3_match_check", "scripts/bulletproof/stage3_match_check.py", []),
        StageDef("stage4_robustness_grid", "scripts/bulletproof/stage4_robustness_grid.py", []),
        StageDef("stage5_reliability", "scripts/bulletproof/stage5_reliability.py", []),
        StageDef("stage6_clinical_stability", "scripts/bulletproof/stage6_clinical_stability.py", []),
        StageDef("stage7_optional_bio", "scripts/bulletproof/stage7_optional_bio.py", []),
        StageDef("stage8_packaging", "scripts/bulletproof/stage8_packaging.py", []),
        StageDef("stage9_tarball", "scripts/bulletproof/stage9_tarball.py", []),
    ]

    run_records: List[Dict[str, Any]] = []
    fail_hard = False

    for st in stages:
        st_extra: Dict[str, Optional[str]] = {
            "data_root": str(args.data_root),
            "wall_hours": str(args.wall_hours),
            "expected_kit": str(args.expected_kit) if args.expected_kit else "",
            "max_workers": "3",
        }

        if st.name in {"stage0_preflight", "stage3_match_check", "stage8_packaging", "stage9_tarball"}:
            st_extra.pop("data_root", None)
            st_extra.pop("wall_hours", None)
            st_extra.pop("max_workers", None)
        if st.name == "stage1_stage_raw":
            st_extra.pop("wall_hours", None)
            st_extra.pop("max_workers", None)
        if st.name in {"stage4_robustness_grid", "stage6_clinical_stability"}:
            st_extra.pop("expected_kit", None)
            st_extra.pop("wall_hours", None)
        if st.name in {"stage5_reliability", "stage7_optional_bio", "stage8_packaging", "stage9_tarball", "stage3_match_check"}:
            st_extra.pop("expected_kit", None)
        if st.name == "stage2_repro_confirmatory":
            st_extra.pop("expected_kit", None)

        if st.name == "stage2_repro_confirmatory" and (audit / "FORCE_STAGE2_LOCAL").exists():
            st.args = list(st.args) + ["--force_local"]

        ok = _run_stage(
            st,
            repo_root,
            out_root,
            {k: v for k, v in st_extra.items() if v is not None},
            audit,
            master_log,
            max_attempts=max(1, int(args.max_attempts)),
        )
        run_records.append({"stage": st.name, "ok": ok})

        if not ok and st.name in {"stage0_preflight", "stage1_stage_raw", "stage2_repro_confirmatory", "stage3_match_check"}:
            fail_hard = True
            if st.name in {"stage2_repro_confirmatory", "stage3_match_check"}:
                stop_reason(
                    audit / "STOP_REASON_exhausted_attempts.md",
                    "master",
                    f"Exceeded max attempts for {st.name}.",
                    diagnostics={
                        "stage": st.name,
                        "max_attempts": int(args.max_attempts),
                        "stage_log": str(audit / f"{st.name}.log"),
                    },
                )
            break

    status = {
        "out_root": str(out_root),
        "slurm": slurm,
        "records": run_records,
        "fail_hard": fail_hard,
        "max_attempts": int(args.max_attempts),
    }
    write_json(audit / "run_status.json", status)

    if fail_hard:
        stop_reason(audit / "STOP_REASON_master.md", "master", "Hard failure before completion.", diagnostics=status)
        return 1

    print(f"OUT_ROOT={out_root}")
    print(f"AUDIT_REPORT={audit / 'BULLETPROOF_AUDIT_REPORT.md'}")
    print(f"MANUSCRIPT_ZIP={paths['OUTZIP'] / 'MANUSCRIPT_KIT_UPDATED.zip'}")
    print(f"OVERLEAF_ZIP={paths['OUTZIP'] / 'OVERLEAF_UPDATED.zip'}")
    print(f"PRISM_ZIP={paths['OUTZIP'] / 'PRISM_ARTIST_PACK.zip'}")
    print(f"RESULTS_TARBALL={paths['TARBALLS'] / 'results_only.tar.gz'}")
    print(f"RESULTS_TARBALL_SHA={paths['TARBALLS'] / 'results_only.tar.gz.sha256'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
