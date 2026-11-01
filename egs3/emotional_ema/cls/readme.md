# Emotional EMA speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Emotional EMA alone and scores it on Emotional EMA's own test split, classifying each
utterance into one of five emotions.

[Emotional EMA](https://zenodo.org/records/19303746) is an American English
acted corpus recorded with electromagnetic articulography: 3 speakers (`jn`, `jr`,
`kf`) read sentences and passages in five emotions, `angry`, `fear`, `happy`,
`neutral` and `sad`. The wav files of the `EMA_5EMO_NSF` release number 953,
1.63 hours in total, 1.4 to 63 seconds long. Some neutral clips are fast-rate
variants or repeats (`_rep1`). The speaker IDs match those of the USC-EMO-MRI
recipe.

The Zenodo record describes a different recording (680 utterances, four target
emotions); the archive holds five emotions and 953 recordings, and the numbers
here are those of the archive.

The label is the final label of the listener evaluation, not the emotion the
speaker was asked to act. Each archive directory holds a `<group>_best_emo` table
of `<utterance> <code>` lines; `dataset/config.yaml` maps the codes 1 to 5 to
`neutral`, `angry`, `happy`, `sad` and `fear` (the archive does not explain the
codes; the order is read from its per-listener evaluation file). The final label
differs from the emotion in the file name
`ema_5emo_<speaker>_<emotion>_<rate>_<prompt>_utt<N>.wav` for 68 of the 953 clips.
The archive also holds each listener's votes and ratings, which this recipe does
not use. Three zero-content dummy files do not match the file name pattern and
are skipped.

| label | clips |
|---|---|
| neutral | 277 |
| angry | 182 |
| sad | 175 |
| happy | 160 |
| fear | 159 |

The recordings are 22.05 kHz mono, so `create_dataset` converts them to 16 kHz with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EMOTIONAL_EMA` — the directory holding the extracted `EMA_5EMO_NSF/` wav files and `*_best_emo` tables, or where to download them to
- `EMOTIONAL_EMA_OUTPUT` — where the converted audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Zenodo record 19303746 (`EMA_5EMO_NSF`, downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Zenodo, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 956 | 3 | 1.63 |

How these were counted: every `.wav` in the archive (22.05 kHz mono, 1.4 to 63 s; 3 of the 956 are zero-content dummy files that do not match the file name pattern); hours from the file lengths. Speakers are `jn`, `jr` and `kf` (file names); the archive's README says they are the same people as speakers F1, F2 and M1 of USC-EMO-MRI.

### Files per label

| `best_emo` code in the release | read as | samples |
|---|---|---|
| 1 | neutral | 277 |
| 2 | angry | 182 |
| 3 | happy | 160 |
| 4 | sad | 175 |
| 5 | fear | 159 |

How these were counted: the `<utterance> <code>` lines of the `<group>_best_emo` tables, which hold the final label of the listener evaluation (953 lines, one per non-dummy file). The archive does not explain the codes; the order is read from its per-listener evaluation file.

### Issues in the release

- The Zenodo record describes a different recording (680 utterances, four target emotions, 10 female and 14 male sentences); the archive holds five emotions, 953 utterances, 8 sentences and 2 passages. The record's DOI points to a 2005 paper.
- The final label differs from the emotion in the file name for 68 of the 953 files (885, or 92.9%, agree). The README says the audio is 16 kHz; it is 22.05 kHz.
- In the evaluation file the target emotion column says `anger` for 13 `neutral` passages of speaker `jn`; the listeners' votes for them are `neutral`.
- 33 utterances are longer than 30 s (passages).


## Download

`prepare_source` downloads `EMA_5EMO_NSF.zip` (548 MB, md5-checked) from
[Zenodo](https://zenodo.org/records/19303746) and extracts the wav files and the `*_best_emo` tables. Measured on one run: about 27 minutes to download (about 290 KB/s);
`create_dataset`, `remove_long_short` and `prepare_labels` together took about 72 seconds.

## Splits

The release publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). With only three speakers and shared sentences, the
splits hold the same speakers and prompts, so scores are optimistic.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 667 | 641 |
| valid | 96 | 93 |
| test | 190 | 190 (not filtered) |

`remove_long_short` drops the 29 train and valid clips longer than 30 s.

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
