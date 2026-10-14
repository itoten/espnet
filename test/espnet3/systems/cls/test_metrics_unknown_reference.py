"""Tests for scoring references the model has no class for."""

from pathlib import Path

import pytest

from espnet3.systems.cls.metrics.auc import AUC
from espnet3.systems.cls.metrics.macro_f1 import MacroF1
from espnet3.systems.cls.metrics.mean_ap import MAP
from espnet3.systems.cls.metrics.scoring_utils import (
    build_target_matrix,
    check_unknown_reference,
    drop_unknown_rows,
    resolve_classes,
)
from espnet3.systems.cls.metrics.ua import UA

pytest.importorskip("sklearn")

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_check_unknown_reference_rejects_typo   | Only error/score/skip pass.  |
# | test_resolve_classes_score_adds_unknown     | "score" keeps the label as a |
# |                                             | class nothing predicts.      |
# | test_resolve_classes_error_is_default       | Old behaviour is unchanged.  |
# | test_drop_unknown_rows_skip_filters_all     | "skip" drops the rows from   |
# |                                             | every aligned list.          |
# | test_drop_unknown_rows_other_modes_keep     | "error"/"score" keep rows.   |
# | test_build_target_matrix_allow_unknown      | Unknown label = all-zero row.|
# | test_ua_and_macro_f1_modes                  | score counts it as 0; skip   |
# |                                             | ignores it; error raises.    |
# | test_map_and_auc_modes                      | Unknown rows are negatives   |
# |                                             | ("score") or dropped.        |

TOKENS = ["joy", "anger", "<unk>"]
# utt3 carries a label the model cannot predict. utt1 and utt2 are right.
REF = ["utt1 joy", "utt2 anger", "utt3 contempt", "utt4 joy"]
HYP = ["utt1 joy", "utt2 anger", "utt3 joy", "utt4 joy"]
SCORE = [
    "utt1 0.9 0.1",
    "utt2 0.2 0.8",
    "utt3 0.8 0.2",
    "utt4 0.7 0.3",
]


@pytest.fixture()
def token_list(tmp_path: Path) -> Path:
    path = tmp_path / "token_list"
    path.write_text("\n".join(TOKENS), encoding="utf-8")
    return path


@pytest.fixture()
def inputs(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for name, lines in (("ref", REF), ("hyp", HYP), ("score", SCORE)):
        path = tmp_path / f"{name}.scp"
        path.write_text("\n".join(lines), encoding="utf-8")
        paths[name] = path
    return paths


def test_check_unknown_reference_rejects_typo():
    with pytest.raises(ValueError, match="unknown_reference must be one of"):
        check_unknown_reference("ignore")


def test_resolve_classes_score_adds_unknown(token_list: Path):
    classes = resolve_classes(["joy", "contempt"], token_list, "score")
    assert classes == ["joy", "contempt"]


def test_resolve_classes_error_is_default(token_list: Path):
    with pytest.raises(ValueError, match="not in the token list"):
        resolve_classes(["joy", "contempt"], token_list)


def test_drop_unknown_rows_skip_filters_all(token_list: Path):
    refs, (hyps,) = drop_unknown_rows(
        ["joy", "contempt", "anger"], [["a", "b", "c"]], token_list, "skip"
    )
    assert refs == ["joy", "anger"]
    assert hyps == ["a", "c"]


@pytest.mark.parametrize("mode", ["error", "score"])
def test_drop_unknown_rows_other_modes_keep(token_list: Path, mode: str):
    refs, (hyps,) = drop_unknown_rows(
        ["joy", "contempt"], [["a", "b"]], token_list, mode
    )
    assert refs == ["joy", "contempt"]
    assert hyps == ["a", "b"]


def test_build_target_matrix_allow_unknown():
    target = build_target_matrix(["joy", "contempt"], ["joy", "anger"], True)
    assert target.tolist() == [[1.0, 0.0], [0.0, 0.0]]


def test_ua_and_macro_f1_modes(inputs, token_list: Path):
    here = inputs["ref"].parent
    for cls in (UA, MacroF1):
        with pytest.raises(ValueError, match="not in the token list"):
            cls(token_list=str(token_list))(inputs, "t", here)
    # skip: joy and anger are both perfect on the remaining utterances.
    assert UA(token_list=str(token_list), unknown_reference="skip")(
        inputs, "t", here
    ) == {"UA": 100.0}
    # score: contempt has recall 0, so UA = (1 + 1 + 0) / 3.
    assert UA(token_list=str(token_list), unknown_reference="score")(
        inputs, "t", here
    ) == {"UA": 66.67}
    # score: joy precision 2/3 (utt3 is predicted joy), recall 1 -> F1 0.8;
    # anger F1 1; contempt F1 0.
    assert MacroF1(token_list=str(token_list), unknown_reference="score")(
        inputs, "t", here
    ) == {"MacroF1": 60.0}


def test_map_and_auc_modes(inputs, token_list: Path):
    here = inputs["ref"].parent
    for cls in (MAP, AUC):
        with pytest.raises(ValueError, match="not in the token list"):
            cls(token_list=str(token_list))(inputs, "t", here)
    # skip: scores rank joy and anger correctly on utt1, utt2, utt4.
    assert AUC(token_list=str(token_list), unknown_reference="skip")(
        inputs, "t", here
    ) == {"AUC": 100.0}
    # score: utt3 is a negative for both classes but ranks above utt4 for
    # joy, so the joy ranking is no longer perfect.
    auc = AUC(token_list=str(token_list), unknown_reference="score")(inputs, "t", here)[
        "AUC"
    ]
    assert auc < 100.0
    assert (
        MAP(token_list=str(token_list), unknown_reference="score")(inputs, "t", here)[
            "mAP"
        ]
        < 100.0
    )
