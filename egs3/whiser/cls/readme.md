# WHiSER speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on WHiSER alone and scores it on WHiSER's own test split, classifying each
utterance into one of eight emotions.

WHiSER is a corpus of 5,427 English podcast segments from the MSP lab (UT Dallas),
annotated with a consensus emotion class and arousal / valence / dominance scores.
All files are mono WAV (5,255 at 44.1 kHz and 172 at 22.05 kHz), 3.01 to 10.99
seconds long, 6.35 hours in total. Speaker ids and gender are `Unknown` for every file.

The label is the consensus class `EmoClass` of `labels_consensus.csv`
(`A` angry, `C` contempt, `D` disgust, `F` fear, `H` happy, `N` neutral, `S` sad,
`U` surprise). As in the msppodcast recipe, `O` (free-text other, 47 files) and `X`
(no agreed label, 916 files) are not emotions and are dropped, leaving 4,464
utterances. The label counts are very uneven: neutral 3,492, happy 345, angry 293,
sad 284, surprise 33, contempt 10, fear 6, disgust 1 (before dropping, over all files).

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `WHISER` — the directory holding `wav/` and `labels_consensus.csv`, or where to download them to
- `WHISER_OUTPUT` — where the converted 16 kHz audio and manifests are written

## Download

`prepare_source` fetches the label table from the corpus GitHub repository and the
5,427 WAV files from the lab's public directory listing (no login, no archive),
with 4 parallel workers. Measured on one run, the audio download took about 16
minutes (3.7 GB); `create_dataset`, `remove_long_short` and `prepare_labels`
together took about 3 minutes 47 seconds. The licence has not been confirmed;
check the terms on the MSP lab site before redistributing.

## Splits

The release publishes no partition (`Split_Set` is `NA`). Files sharing the
prefix before the first `.` of the file name (1,089 prefixes, e.g. `001-105`) are
kept in the same split, as a guess at "same source recording" that was not checked
against the corpus metadata. Prefixes are shuffled with `split_seed` and filled
to 70/10/20.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 3,047 | 3,047 |
| valid | 461 | 461 |
| test | 956 | 956 (not filtered) |

Because of the class imbalance, the rare classes are not in every split: valid has
no contempt or disgust, and test has no disgust. Macro-F1 over the classes present
in the test labels is therefore dominated by a few samples for contempt, fear
and surprise.

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
