# Zero-shot Qwen2.5-Omni on multidataset_ser_01 (VoxEmo protocol)

A zero-shot baseline for the `cls` recipe of this experiment: Qwen2.5-Omni-7B,
as released, labels the evaluation corpora without any training on them. The
prompts, decoding, parsing and scores follow VoxEmo (Zhang et al.,
"VoxEmo: Benchmarking Speech Emotion Recognition with Speech LLMs",
arXiv:2603.08936).

## Data

The evaluation corpora and their rows are those of `../cls`: the filtered
manifests that its `create_dataset` and `remove_long_short` stages write under
`$MULTIDATASET_SER_01_OUTPUT`. Run those stages there first. Each corpus is
prompted with its own closed label set: the seven labels of the experiment
that occur in it.

`MULTIDATASET_SER_01_MAX_UTTERANCES` keeps at most that many utterances per
corpus, drawn uniformly at random with `seed` (`conf/inference.yaml`); the
label set still comes from the whole corpus.

## Protocol

| | |
|---|---|
| Model | `Qwen/Qwen2.5-Omni-7B`, text output only (no talker), bf16 |
| Prompts | VoxEmo's five hard-label prompts: Direct, +T (ASR transcript), +A (acoustic caption), +T+A, +T+A+R (reasoning), built from the blocks of its Table 3 (`src/voxemo.py`) |
| Conversation | One user turn, audio then prompt; the chat template's default system turn |
| Decoding | Greedy, at most 512 new tokens |
| Parsing | The `FINAL_LABEL` field, matched case-insensitively against the corpus's label set; an answer that does not match counts as wrong |
| Ensemble | VoxEmo's Eq. 2: each prompt votes for its label, a failed one gives `1/C` to every class; the most likely label is the ensemble's prediction |
| Scores | UA, WA, Micro-F1, Macro-F1 per prompt and for the ensemble, and each prompt's parse failure rate (`src/metrics.py`) |

VoxEmo's toolkit is not public, so what the paper leaves open is assumed and
listed at the top of `src/voxemo.py`: how the blocks are joined, that the
Direct block belongs to the Direct prompt only, how `{labels}` is written, how
the field is stripped, and how an ensemble tie is broken. VoxEmo's soft-label
evaluation is not run: these corpora carry no per-annotator ratings.

Qwen2.5-Omni's training data is not disclosed. MELD, and possibly other
public English corpora, may have been seen in training.

## Running

```
python run.py --stages infer measure \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

Needs `torchvision` (the Qwen2.5-Omni processor builds its video processor
even for audio). Each GPU process takes the test sets named in
`MULTIDATASET_SER_01_TEST_SETS` (all by default), so one process per GPU can
share the corpora; run `measure` once all of them have finished.

Outputs, under `exp/<exp_tag>/inference/<corpus>/`:

| File | Content |
|---|---|
| `ref.scp` | Reference label |
| `hyp.scp` | Ensemble label |
| `hyp_<prompt>.scp` | Label each prompt named, or `<invalid>` |
| `score.scp` | Ensemble distribution, as JSON |
| `answer.scp` | Raw answers per prompt, as JSON |

and the scores in `exp/<exp_tag>/inference/metrics.json`.
