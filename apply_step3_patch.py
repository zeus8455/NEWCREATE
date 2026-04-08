from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Pattern not found while patching {label}")
    return text.replace(old, new, 1)


def patch_models(repo_root: Path) -> None:
    path = repo_root / "pokervision" / "models.py"
    text = _read(path)

    if "amount_state: Dict[str, Any] = field(default_factory=dict)" not in text:
        text, count = re.subn(
            r'(\n\s*table_amount_state: Dict\[str, Any\] = field\(default_factory=dict\)\n)(\s*action_inference: Dict\[str, Any\] = field\(default_factory=dict\))',
            r'\1\n    amount_state: Dict[str, Any] = field(default_factory=dict)\n\n    amount_normalization: Dict[str, Any] = field(default_factory=dict)\n\n    \2',
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError('Pattern not found while patching models.FrameAnalysis fields')

    if '"amount_state": dict(self.amount_state),' not in text:
        text, count = re.subn(
            r'(\n\s*"table_amount_state": dict\(self\.table_amount_state\),\n)(\s*"action_inference": dict\(self\.action_inference\),)',
            r'\1\n            "amount_state": dict(self.amount_state),\n\n            "amount_normalization": dict(self.amount_normalization),\n\n            \2',
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError('Pattern not found while patching models.FrameAnalysis.to_dict')

    _write(path, text)


def patch_pipeline(repo_root: Path) -> None:
    path = repo_root / "pokervision" / "pipeline.py"
    text = _read(path)

    if "normalized_amount_state = normalize_amount_contributions(" not in text:
        text = _replace_once(
            text,
            "        analysis.amount_normalization = normalize_amount_contributions(",
            "        normalized_amount_state = normalize_amount_contributions(",
            label="pipeline normalize call",
        )

    if "analysis.amount_state = normalized_amount_state" not in text:
        text = _replace_once(
            text,
            "        if analysis.amount_normalization.get(\"warnings\"):\n            analysis.warnings.extend(list(analysis.amount_normalization.get(\"warnings\", [])))",
            "        analysis.amount_state = normalized_amount_state\n        analysis.amount_normalization = normalized_amount_state\n        if normalized_amount_state.get(\"warnings\"):\n            analysis.warnings.extend(list(normalized_amount_state.get(\"warnings\", [])))",
            label="pipeline amount_state assign",
        )

    _write(path, text)


def patch_hand_state(repo_root: Path) -> None:
    path = repo_root / "pokervision" / "hand_state.py"
    text = _read(path)

    text = _replace_once(
        text,
        '            actions_log=list(analysis.action_inference.get("actions_this_frame", [])),',
        '            actions_log=list(analysis.action_inference.get("action_history", analysis.action_inference.get("actions_this_frame", []))),',
        label="hand_state create_hand actions_log",
    )
    text = _replace_once(
        text,
        '        hand.actions_log.extend(list(analysis.action_inference.get("actions_this_frame", [])))',
        '        hand.actions_log = list(analysis.action_inference.get("action_history", analysis.action_inference.get("actions_this_frame", [])))',
        label="hand_state update actions_log",
    )
    text = _replace_once(
        text,
        '            "action_summary": list(analysis.action_inference.get("actions_this_frame", [])),',
        '            "action_summary": list(analysis.action_inference.get("action_history", analysis.action_inference.get("actions_this_frame", []))),',
        label="hand_state frame log action_summary",
    )

    _write(path, text)


def copy_replacements(repo_root: Path) -> None:
    src_action = ROOT / "pokervision" / "action_inference.py"
    dst_action = repo_root / "pokervision" / "action_inference.py"
    shutil.copyfile(src_action, dst_action)

    tests_dir = repo_root / "pokervision" / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "pokervision" / "tests" / "test_action_reconstruction.py", tests_dir / "test_action_reconstruction.py")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Apply PokerVision NEWCREATE step 3 patch")
    parser.add_argument("repo_root", nargs="?", default=".", help="Path to NEWCREATE repository root")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / "pokervision").exists():
        raise SystemExit(f"pokervision folder not found in {repo_root}")

    copy_replacements(repo_root)
    patch_models(repo_root)
    patch_pipeline(repo_root)
    patch_hand_state(repo_root)
    print("Step 3 patch applied successfully.")


if __name__ == "__main__":
    main()
