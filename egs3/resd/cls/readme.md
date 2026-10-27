# RESD speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on RESD alone and scores it on RESD's own test split, classifying each
utterance into one of seven emotions.

[RESD](https://huggingface.co/datasets/Aniemore/resd) (Russian Emotional
Speech Dialogs) is Russian speech recorded in a studio by 20 actors in pairs.
The dialogues were improvised: each actor was privately given an emotion to
play, so the words are spontaneous and the label is the emotion the actor was
asked to convey. 1396 utterances in seven emotions: `fear` (223), `angry`
(219), `happy` (218), `enthusiasm` (198), `neutral` (191), `disgust` (185) and
`sad` (162).

## The corpus is a dialogue corpus

This is the most useful property of RESD: the utterances are turns of 50 improvised
dialogues between two actors, not isolated sentences, so analyses that use the
conversation context are possible (the other speaker's utterance before or after, how
an emotion answers another emotion, the pair of emotions of a dialogue).

The structure is in the name, `<dialog>_<emotion 1>_<emotion 2>_<role>_<index>` (e.g.
`01_happiness_anger_a_020`): the dialogue number (01 to 50), the two emotions given to
the two actors of the dialogue, the role (the first letter of the emotion of the actor
who is speaking: `a` anger, `d` disgust, `e` enthusiasm, `f` fear, `h` happiness, `n`
neutral, `s` sadness; this matches the `emotion` label of all 1396 utterances) and a number
that counts the exchanges (10, 20, 30, ...; both actors have an utterance with the same
number). The numbering suggests the order of the turns but this was not checked
against the audio. The name does not identify the actor.

The recipe classifies each utterance on its own and does not use the context. The
official train/test split is not by dialogue: all 50 dialogues have utterances in both
splits, so neighboring turns of a test utterance can be in the training set.

The release is one parquet file per official split on Hugging Face; each row
holds the recording as WAV bytes, its `emotion` and its `name`,
`<dialog>_<emotion>_<emotion>_<role>_<index>`. The label comes from the
`emotion` column, renamed (`anger` to `angry`, `happiness` to `happy`,
`sadness` to `sad`). 301 names have a
space before the role letter, which is replaced with `_` in the utterance id.
The recordings are mono, 716 at 16 kHz and 680 at 44.1 kHz, so
`create_dataset` resamples all of them to 16 kHz with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `RESD` — a directory holding the two parquet files (`data/*.parquet`), or
  where to download them to
- `RESD_OUTPUT` — where the resampled audio and manifests are written

## Download

`prepare_source` downloads the train and test parquet files (485 MB in total)
from [Hugging Face](https://huggingface.co/datasets/Aniemore/resd), pinned to
one revision in `dataset/config.yaml`. The dataset is not gated. Its dataset
card gives the license as MIT.

## Splits

The official train/test split is kept. No validation split is published, so
10% of the official train is drawn at random within each label (`valid_ratio`,
`split_seed` in `dataset/config.yaml`). The official split is not speaker
independent: all 50 dialogs appear in both train and test.

| split | utterances |
|---|---|
| train | 1004 |
| valid | 112 |
| test | 280 |

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
