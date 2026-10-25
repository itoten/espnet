# LEGOv2 anger-level recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on LEGOv2 alone and scores it on LEGOv2's own test split, classifying each
user utterance into one of five anger levels. This is not a general emotion
corpus: every utterance is a caller speaking to a bus-information dialogue system.

LEGOv2 is a parameterised and annotated version of the CMU Let's Go
spoken-dialogue data (calls recorded in 2006 and 2007, English), prepared by the
Dialogue Systems Group of Ulm University. The release has 548 calls and 13,836
system-user exchanges. One annotator labelled the emotional state of the user's
utterance in 302 calls; the labels follow the scheme `friendly`, `neutral`,
`slightly angry`, `angry`, `very angry`, stored in the `EmotionalState` column of
`interactions.csv`. 4,832 utterances carry a label and a recording. 589 of them are
labelled `garbage`, a value that is not in the annotation scheme and whose meaning
is not documented. Listening to a sample of them (24 clips drawn at random, a few
of them played) showed silence or clips that no emotion could be assigned to
(median length 1.0 s, mostly one-word replies, key presses or empty recognition
results), so the recipe leaves them out. The remaining 4,243 utterances from
291 calls last 1.97 hours in total, 0.58 to 21.7 seconds each. The recordings are
8 kHz mono telephone speech, so `create_dataset` converts them to 16 kHz with
ffmpeg (upsampling).

| label | utterances |
|---|---|
| neutral | 3,305 |
| slightly_angry | 693 |
| angry | 137 |
| very_angry | 104 |
| friendly | 4 |

`friendly` has only 4 utterances, so it is effectively untrainable and untestable;
it is kept as a class anyway.
The corpus has no speaker id; callers are anonymous and the same person may have
called more than once.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `LEGOV2` — the directory holding the extracted `LEGOv2/` directory, or where to download it to
- `LEGOV2_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `LEGOv2.zip` (1.19 GB, no login) from
`https://data.ds.uni-bamberg.de/LEGO/LEGOv2.zip`, checks its SHA-256 (computed
from the file; the university publishes no checksum) and extracts the csv, the
licence and the wav files. The download took about 12 minutes on one run. Then
`create_dataset`, `remove_long_short` and `prepare_labels` took 3 minutes 36
seconds including extraction and the SHA-256 check.

## Licence and citation

`license.txt` in the archive grants use "for individual, education research
purposes only" for both the CMU recordings and the Ulm University annotations.
It says nothing on redistribution, so do not redistribute the data. Cite both:

- A. Schmitt, S. Ultes and W. Minker, "A Parameterized and Annotated Spoken Dialog
  Corpus of the CMU Let's Go Bus Information System", LREC 2012, pp. 3369-3373.
- S. Ultes, A. Schmitt, M. J. Platero Sanchez and W. Minker, "Analysis of an
  Extended Interaction Quality Corpus", IWSDS 2015.

## Splits

The corpus publishes no partition. Whole calls are kept in one split, since the
utterances of a call come from one caller. Calls are assigned greedily so that
each split's label proportions approach 70 / 10 / 20 (`train_ratio`,
`valid_ratio`, `test_ratio`, `split_seed` in `dataset/config.yaml`). Callers are
anonymous, so a caller who made several calls may appear in more than one split.

| split | calls | utterances | after `remove_long_short` | hours | neutral | slightly_angry | angry | very_angry | friendly |
|---|---|---|---|---|---|---|---|---|---|
| train | 246 | 2,887 | 2,887 | 1.29 | 2,314 | 429 | 73 | 68 | 3 |
| valid | 13 | 452 | 452 | 0.23 | 330 | 93 | 23 | 6 | 0 |
| test | 32 | 904 | 904 (not filtered) | 0.45 | 661 | 171 | 41 | 30 | 1 |

The valid split has only 6 `very_angry` and no `friendly` utterances.

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
