# ESD speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on ESD alone and scores it on ESD's own test split, classifying each
utterance into one of five emotions.

[ESD](https://github.com/HLTSingapore/Emotional-Speech-Data) (Emotional Speech
Dataset; Zhou, Sisman, Liu and Li, Speech Communication 2022) is acted speech from
20 speakers, 10 reading Mandarin and 10 reading English, who each speak 350
parallel sentences in `neutral`, `happy`, `angry`, `sad` and `surprise`. There
are 35,000 recordings, 29.1 hours in total, each 1.1 to 8.1 seconds long, with
exactly 7,000 per emotion. There is no `disgust` or `fear`.

The label comes from the emotion folder, `<speaker>/<Emotion>/<speaker>_<item>.wav`
(e.g. `0003/Happy/0003_000898.wav`); `create_dataset` fails if the speaker folder
disagrees with the file name. Speakers 0001-0010 are Mandarin and 0011-0020 are
English. The recordings are already 16 kHz mono 16-bit; `create_dataset` still
passes them through ffmpeg, `num_workers` files at a time.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `ESD` — the directory holding the extracted audio, or where to download it to
- `ESD_OUTPUT` — where the converted audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the ESD (Emotional Speech Dataset) archive from the project's Google Drive link (`HLTSingapore/Emotional-Speech-Data`; 2.4 GB; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Download from the project's Google Drive link. The project asks users to fill in a license form and send it to the authors; whether the form was sent before this download is not recorded here. The archive has no checksum (completeness was checked by the 35,000 file count).

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 35,000 | 20 | 29.1 |

How these were counted: every `.wav` under `<speaker>/<Emotion>/` (16 kHz mono 16-bit, 1.14 to 8.08 s); hours from the file lengths. Speakers are the 20 speaker folders: 0001 to 0010 read Mandarin and 0011 to 0020 read English (five women and five men per language); 350 parallel sentences per speaker and emotion.

### Files per label

| label (emotion folder in the release) | samples | hours |
|---|---|---|
| Neutral | 7,000 | 5.659 |
| Happy | 7,000 | 5.380 |
| Angry | 7,000 | 5.328 |
| Sad | 7,000 | 6.826 |
| Surprise | 7,000 | 5.878 |

How these were counted: the emotion folder each file is in. The release has no disgust or fear class. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- No byte-identical, decoded-identical or near-identical recordings were found.


## Download

`prepare_source` downloads the Google Drive file linked from the authors' GitHub
page (2.4 GB) and extracts only the WAV files. No checksum is published, so
completeness is checked by counting the 35,000 files. The GitHub page asks users
to complete a license form and send it to the authors, and states that the data
may be used for research only.

## Splits

The archive publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out, so the splits share all
20 speakers. The 350 sentences are shared across emotions (and, within a
language, mostly across speakers), so the same sentence appears in all splits.

| split | utterances |
|---|---|
| train | 24500 |
| valid | 3500 |
| test | 7000 |

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
