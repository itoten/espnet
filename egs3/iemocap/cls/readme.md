# IEMOCAP speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on IEMOCAP alone and scores it on IEMOCAP's own test split, classifying
each utterance into one of six emotions.

[IEMOCAP](https://sail.usc.edu/iemocap/) is English dyadic conversation,
scripted and improvised, recorded in five sessions of two actors each. The
labels are read from the release's own `Session*/dialog/EmoEvaluation/*.txt`,
and the audio from `Session*/sentences/wav/`, which is already 16 kHz mono.

Six of the eleven label codes are used, and `hap` and `exc` stay separate
classes rather than being merged as in the common four-class setup:

| code | label | utterances |
|---|---|---|
| `neu` | `neutral` | 1708 |
| `ang` | `angry` | 1103 |
| `sad` | `sad` | 1084 |
| `exc` | `excited` | 1041 |
| `hap` | `happy` | 595 |
| `fea` | `fear` | 40 |

The other codes (`fru`, `sur`, `dis`, `oth`, `xxx`) are dropped, leaving 5571
of the 10039 utterances. Both scripted and improvised dialogs are used.

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
| train | 3899 |
| valid | 557 |
| test | 1115 |

One training utterance is longer than 30 s and is removed by
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
