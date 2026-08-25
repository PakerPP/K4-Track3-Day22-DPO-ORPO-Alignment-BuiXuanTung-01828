"""CPU-only smoke tests — run without a GPU (no torch/unsloth/trl import).

These guard the lab source against the most common breakages so `make test`
is a real gate, not a no-op:
- every notebook/script file exists and is valid Python (catches syntax errors)
- the TRL trainer calls use `processing_class=` (TRL >= 0.13), NOT the removed
  `tokenizer=` arg — the regression that broke NB1/NB3 on the resolved trl 0.19.x

Run:  pytest -q scripts/   (or `make test`).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    "01_sft_mini", "02_preference_data", "03_dpo_train",
    "04_compare_and_eval", "05_merge_deploy_gguf", "06_benchmark",
]


def test_notebooks_exist_and_parse():
    for nb in NOTEBOOKS:
        p = REPO / "notebooks" / f"{nb}.py"
        assert p.exists(), f"missing notebook {p}"
        ast.parse(p.read_text(encoding="utf-8"))  # SyntaxError if broken


def test_scripts_parse():
    for p in (REPO / "scripts").glob("*.py"):
        ast.parse(p.read_text(encoding="utf-8"))


def test_colab_notebooks_are_valid_json():
    for p in (REPO / "colab").glob("*.ipynb"):
        json.loads(p.read_text(encoding="utf-8"))  # ValueError if corrupt


def test_trainer_uses_processing_class_not_tokenizer():
    # TRL >= 0.13 removed the `tokenizer=` arg in favour of `processing_class=`.
    # With the requirements pin `trl>=0.12,<0.20` a fresh install resolves to
    # 0.19.x, where `DPOTrainer/SFTTrainer(tokenizer=...)` raises TypeError.
    targets = [
        "notebooks/01_sft_mini.py",
        "notebooks/03_dpo_train.py",
        "scripts/train_dpo.py",
        "colab/Lab22_DPO_T4.ipynb",
        "colab/Lab22_DPO_BigGPU.ipynb",
    ]
    offenders = [t for t in targets if "tokenizer=tokenizer" in (REPO / t).read_text(encoding="utf-8")]
    assert not offenders, (
        f"{offenders} still pass tokenizer=tokenizer to a TRL trainer; "
        f"use processing_class=tokenizer (tokenizer= removed in trl>=0.13)."
    )


def test_dpo_adapter_contract_is_consistent_everywhere():
    """The saved DPO adapter must be a standalone SFT-initialized policy."""
    required = [
        "adapter_name=\"reference\"",
        "adapter_name=\"default\"",
        "model_adapter_name=\"default\"",
        "ref_adapter_name=\"reference\"",
        "selected_adapters=[\"default\"]",
    ]
    for path in [REPO / "notebooks" / "03_dpo_train.py", REPO / "scripts" / "train_dpo.py"]:
        source = path.read_text(encoding="utf-8")
        missing = [needle for needle in required if needle not in source]
        assert not missing, f"{path} is missing DPO adapter-contract settings: {missing}"

    for path in [REPO / "notebooks" / "05_merge_deploy_gguf.py", REPO / "scripts" / "merge_and_gguf.py"]:
        source = path.read_text(encoding="utf-8")
        assert "PeftModel.from_pretrained(model, str(DPO_PATH))" in source or \
               "PeftModel.from_pretrained(model, args.dpo_path)" in source, \
               f"{path} must merge the DPO policy adapter, not SFT-only"

    for path in (REPO / "colab").glob("*.ipynb"):
        source = path.read_text(encoding="utf-8")
        for needle in ["adapter_name=\\\"reference\\\"", "ref_adapter_name=\\\"reference\\\"",
                       "PeftModel.from_pretrained(model, str(DPO_PATH))"]:
            assert needle in source, f"{path} is not synced with the DPO adapter contract"
