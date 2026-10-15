# IESC speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on IESC alone and scores it on IESC's own test split, classifying each
utterance into one of five emotions.

IESC (Indian Emotional Speech Corpora) is an acted English corpus of eight
north Indian speakers (5 male, 3 female according to the Kaggle description),
recorded in a closed room with a speech-recorder app. Each speaker reads two
sentences ("Kids are talking by the door" / "Dogs are sitting by the door")
several times in each emotion. The folder layout is
`Speaker-<n>/<Emotion>/<E>-<take>-<n>-<sentence>.wav`, and the folder names give
the label.

Measured on the copy used here: 600 wav files, 0.56 hours in total, 1.77 to
5.50 seconds long. Emotions: angry 120, fear 120, happy 120, neutral 120, sad 120
(`Anger` is written `angry` here, as in the other recipes). Speakers 1 to 4 have
50 files each and speakers 5 to 8 have 100. The recordings are in mixed formats
as distributed (16, 32, 44.1 or 48 kHz; mono or stereo; all 16-bit) and are
converted to 16 kHz mono.

Things to know when reading the scores:

- Only two sentences exist, so the text carries no emotion information but the
  model sees the same two sentences in every split.
- Two pairs of files are byte-identical copies (`Speaker-3/Sad/S-1-3-1.wav` =
  `S-3-3-1.wav`, `Speaker-8/Anger/A-10-8-2.wav` = `A-7-8-2.wav`). They are kept
  as distributed. One file name looks mistyped (`Speaker-8/Sad/S-8-8-18.wav`,
  where the other takes end in `-1` or `-2`); it is used as is.
- The recording format differs by speaker, and whole speakers are held out, so
  the splits also differ in recording device.

## Provenance

Kaggle dataset `ybsingh/indian-emotional-speech-corpora-iesc` (version 1,
2022-01-28), licence shown on Kaggle: Unknown. The paper cited in the dataset
entry is Singh and Goel, Multimedia Tools and Applications 82, 2023
(doi:10.1007/s11042-023-14577-w); the citation requested by the authors was not
checked.

Kaggle requires a login, so the recipe does not download. Download the zip
(138 MB) and extract it:

```bash
unzip indian-emotional-speech-corpora-iesc.zip -d "$IESC"
```

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `IESC` — the directory holding (or containing, at any depth) the `Speaker-<n>/` folders
- `IESC_OUTPUT` — where the converted audio and manifests are written

## Download

The zip took 6 seconds to download (unzip not timed separately).
`create_dataset`, `remove_long_short` and `prepare_labels` together took about
60 seconds on CPU.

## Splits

No partition is published. Whole speakers are held out: the eight speakers are
shuffled with `split_seed` (`dataset/config.yaml`) and two go to test, one to
valid, five to train. This gives the following splits.

| split | speakers | utterances | after `remove_long_short` |
|---|---|---|---|
| train | 1, 3, 4, 7, 8 | 350 | 350 |
| valid | 6 | 100 | 100 |
| test | 2, 5 | 150 | 150 (not filtered) |

Each speaker has the same number of files per emotion, so the classes are
balanced in every split.

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
