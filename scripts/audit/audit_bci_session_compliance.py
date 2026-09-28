"""Audit entry point for explicit BCI session data-completeness profiles.

The audit is read-only with respect to the user-supplied source tree. It never
imputes absent media or treats a zero placeholder as an observed modality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eptnet.data.prepare_bci_subjects import (
    COMPLIANCE_PROFILES,
    build_session_compliance,
    discover_sessions,
    prepare_session,
    render_session_compliance_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../BCI"))
    parser.add_argument(
        "--profile",
        choices=tuple(COMPLIANCE_PROFILES),
        default="strong_behavior_sources",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/bci_session_compliance.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("results/bci_session_compliance.md"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 when at least one session fails the selected profile.",
    )
    args = parser.parse_args()

    sources = discover_sessions(args.source.expanduser().resolve())
    sessions = [prepare_session(source) for source in sources]
    report = build_session_compliance(sessions, required_profile=args.profile)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    args.markdown_output.write_text(
        render_session_compliance_markdown(report), encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "selected_profile": report["selected_profile"],
                "compliant_sessions": report["compliant_sessions"],
                "total_sessions": report["total_sessions"],
                "all_sessions_compliant": report["all_sessions_compliant"],
                "failed_check_counts": report["selected_profile_failed_check_counts"],
                "json_output": str(args.json_output),
                "markdown_output": str(args.markdown_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 2 if args.strict and not report["all_sessions_compliant"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
