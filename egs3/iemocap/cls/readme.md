# IEMOCAP speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on IEMOCAP alone and scores it on IEMOCAP's own test split, classifying
each utterance into one of eleven classes.

[IEMOCAP](https://sail.usc.edu/iemocap/) is English dyadic conversation,
scripted and improvised, recorded in five sessions of two actors each. The
labels are read from the release's own `Session*/dialog/EmoEvaluation/*.txt`,
and the audio from `Session*/sentences/wav/`, which is already 16 kHz mono.

Every label code in the evaluation files is a class and none is dropped, so that
filtering (rare classes, `no_majority`, ...) can be decided later. `hap` and `exc`
stay separate classes rather than being merged as in the common four-class setup:

| code | label | utterances |
|---|---|---|
| `xxx` | `no_majority` | 2507 |
| `fru` | `frustrated` | 1849 |
| `neu` | `neutral` | 1708 |
| `ang` | `angry` | 1103 |
| `sad` | `sad` | 1084 |
| `exc` | `excited` | 1041 |
| `hap` | `happy` | 595 |
| `sur` | `surprise` | 107 |
| `fea` | `fear` | 40 |
| `oth` | `other` | 3 |
| `dis` | `disgust` | 2 |

All 10039 utterances are used. `xxx` is the release's mark for utterances for
which no majority label could be assigned among the annotators (its README); it is
kept as the class `no_majority`. `disgust` (2) and `other` (3) are far too small to
learn from but are kept. Both scripted and improvised dialogs are used.

## Listening test

Each utterance was evaluated categorically by at least three annotators (and
dimensionally by at least two) from the set angry, happy, sad, neutral, frustrated,
excited, fearful, surprised, disgusted and other (the release's README). The label
here is the majority of these evaluations. The individual evaluations are in the
`EmoEvaluation` files and are not used here.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `IEMOCAP` — a copy of the release holding `Session1` to `Session5`
- `IEMOCAP_OUTPUT` — where the converted audio and manifests are written

## Download

The corpus needs a signed release agreement, requested through
[sail.usc.edu/iemocap](https://sail.usc.edu/iemocap/iemocap_release.htm), so it
cannot be fetched from a script. Point `$IEMOCAP` at the extracted release.

## Splits

Utterances are split 70/10/20 at random within each label (`train_ratio`,
`valid_ratio`, `test_ratio`, `split_seed` in `dataset/config.yaml`). Sessions
and speakers are not held out, so all three splits contain all ten speakers.

| split | utterances |
|---|---|
| train | 7027 |
| valid | 1004 |
| test | 2008 |

Two training utterances are longer than 30 s and are removed by
`remove_long_short`.

## Quick start

Run the three commands separately. With several GPUs, `train` starts one
process per GPU by re-running its own command, so any other stage listed in the
same command would run once per GPU as well.

```bash
python run.py --stages create_dataset remove_long_short prepare_labels \
    collect_stats --training_config conf/training.yaml
python run.py --stages train \
    --training_config conf/training.yaml
python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```
