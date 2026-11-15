"""Tests for the VoxEmo protocol of multidataset_ser_01/qwen2_5_omni."""

import json

import numpy as np
import pytest

from egs3.multidataset_ser_01.qwen2_5_omni.src.inference import INVALID, build_output
from egs3.multidataset_ser_01.qwen2_5_omni.src.metrics import VoxEmoScores
from egs3.multidataset_ser_01.qwen2_5_omni.src.voxemo import (
    PROMPT_BLOCKS,
    VoxEmoClassifier,
    build_prompt,
    ensemble_votes,
    parse_final_label,
)

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                  | Description                   |
# |--------------------------------------------|-------------------------------|
# | test_prompt_direct_is_table3               | Exact text of the Direct      |
# |                                            | prompt.                       |
# | test_prompt_t_a_r_lists_fields_in_order    | Blocks and response fields in |
# |                                            | Table 3 order; no Direct.     |
# | test_five_prompts                          | Direct, +T, +A, +T+A, +T+A+R. |
# | test_parse_final_label                     | Case, brackets, last line,    |
# |                                            | out-of-set and missing field. |
# | test_ensemble_is_eq2                       | Failures vote 1/C each.       |
# | test_classifier_runs_every_prompt          | One generate call per prompt, |
# |                                            | batched; ensemble label.      |
# | test_classifier_single_item                | Unbatched call returns a dict.|
# | test_classifier_rejects_unknown_prompt     | Typos fail early.             |
# | test_build_output_fields                   | ref, hyp, hyp_<prompt>,       |
# |                                            | <invalid>, one-line JSON.     |
# | test_scores                                | UA/WA/F1/parse failure values.|


def test_prompt_direct_is_table3():
    assert build_prompt("direct", ["neutral", "angry"]) == (
        "You are a speech emotion recognition system. You are given an audio clip.\n"
        "Choose exactly one emotion label from this closed set: neutral, angry.\n"
        "Decide the emotion directly from the audio.\n"
        "Response format:\n"
        "FINAL_LABEL: <one label from neutral, angry>"
    )


def test_prompt_t_a_r_lists_fields_in_order():
    lines = build_prompt("t_a_r", ["sad"]).splitlines()

    assert lines[2].startswith("Transcribe the spoken content")
    assert lines[3].startswith("Describe acoustic and paralinguistic cues")
    assert lines[4].startswith("Explain how the transcript content")
    assert lines[5:] == [
        "Response format:",
        "ASR_TRANSCRIPT: <transcript>",
        "ACOUSTIC_CAPTION: <description>",
        "REASONING: <reasoning>",
        "FINAL_LABEL: <one label from sad>",
    ]
    assert "Decide the emotion directly" not in "\n".join(lines)


def test_five_prompts():
    assert list(PROMPT_BLOCKS) == ["direct", "t", "a", "t_a", "t_a_r"]


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("FINAL_LABEL: angry", "angry"),
        ("FINAL_LABEL: ANGRY", "angry"),
        ("FINAL_LABEL: <Sad>.", "sad"),
        ("FINAL_LABEL: **neutral**", "neutral"),
        ("ASR_TRANSCRIPT: hi\nREASONING: calm\nFINAL_LABEL: neutral", "neutral"),
        ("FINAL_LABEL: angry\nFINAL_LABEL: sad", "sad"),
        ("FINAL_LABEL: happy", None),  # not in this corpus's set
        ("FINAL_LABEL: angry or sad", None),
        ("The speaker sounds angry.", None),
        ("", None),
    ],
)
def test_parse_final_label(answer, expected):
    assert parse_final_label(answer, ["neutral", "sad", "angry"]) == expected


def test_ensemble_is_eq2():
    distribution = ensemble_votes(["sad", None, "sad", "angry", None], ["angry", "sad"])

    np.testing.assert_allclose(distribution, [(1 + 2 * 0.5) / 5, (2 + 2 * 0.5) / 5])
    np.testing.assert_allclose(ensemble_votes([None] * 5, ["a", "b"]), [0.5, 0.5])


class FakeGenerator:
    """Answers by prompt: Direct -> angry, +T -> unparsable, others -> sad."""

    def __init__(self, device):
        self.device = device
        self.calls = []

    def generate(self, speech, prompts):
        self.calls.append((len(speech), prompts))
        answers = []
        for prompt in prompts:
            if "directly" in prompt:
                answers.append("FINAL_LABEL: angry")
            elif "REASONING" not in prompt and "ACOUSTIC" not in prompt:
                answers.append("ASR_TRANSCRIPT: hello")
            else:
                answers.append("ACOUSTIC_CAPTION: low\nFINAL_LABEL: Sad")
        return answers


def test_classifier_runs_every_prompt():
    model = VoxEmoClassifier(generator=FakeGenerator, device="cuda:3")

    results = model(speech=[np.zeros(4), np.zeros(4)], label_set=["angry, sad"] * 2)

    assert model.generator.device == "cuda:3"
    assert [n for n, _ in model.generator.calls] == [2] * 5
    first = results[0]
    assert first["labels"] == ["angry", "sad"]
    assert first["predictions"] == {
        "direct": "angry",
        "t": None,
        "a": "sad",
        "t_a": "sad",
        "t_a_r": "sad",
    }
    np.testing.assert_allclose(first["distribution"], [1.5 / 5, 3.5 / 5])
    assert first["prediction"] == "sad"


def test_classifier_single_item():
    model = VoxEmoClassifier(generator=FakeGenerator, variants=["direct"])

    result = model(speech=np.zeros(4), label_set="neutral, angry")

    assert result["prediction"] == "angry"
    assert list(result["answers"]) == ["direct"]


def test_classifier_rejects_unknown_prompt():
    with pytest.raises(ValueError, match="direkt"):
        VoxEmoClassifier(generator=FakeGenerator, variants=["direkt"])


def test_build_output_fields():
    output = {
        "answers": {"direct": "FINAL_LABEL: sad\n", "t": "oops"},
        "predictions": {"direct": "sad", "t": None},
        "labels": ["angry", "sad"],
        "distribution": np.array([0.25, 0.75]),
        "prediction": "sad",
    }
    item = {"utt_id": "u1", "label": "angry"}

    records = build_output(data=[item], model_output=[output], idx=[0])

    record = records[0]
    assert record["utt_id"] == "u1"
    assert record["ref"] == "angry"
    assert record["hyp"] == "sad"
    assert record["hyp_direct"] == "sad"
    assert record["hyp_t"] == INVALID
    assert json.loads(record["score"]) == {"angry": 0.25, "sad": 0.75}
    assert "\n" not in record["answer"]
    assert json.loads(record["answer"])["direct"] == "FINAL_LABEL: sad\n"
    assert build_output(data=item, model_output=output, idx=0) == record


def test_scores():
    metric = VoxEmoScores(variants=["direct"])
    metric.update(
        ref=["a", "a", "b", "b"],
        hyp=["a", "a", "b", "a"],
        hyp_direct=["a", INVALID, "b", INVALID],
    )

    result = metric.compute()

    assert result["WA_ensemble"] == 75.0
    assert result["UA_ensemble"] == 75.0  # recall a=1, b=0.5
    assert result["WA_direct"] == 50.0
    assert result["UA_direct"] == 50.0  # recall a=0.5, b=0.5
    assert result["ParseFailure_direct"] == 50.0
    # Micro-F1 over the classes: precision 2/2, recall 2/4.
    assert result["MicroF1_direct"] == pytest.approx(66.67)
    assert result["MacroF1_direct"] == pytest.approx(66.67)
    assert "ParseFailure_ensemble" not in result
