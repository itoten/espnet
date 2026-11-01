# SITB-OSED speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on SITB-OSED alone and scores it on its own test split, classifying each
utterance into one of six emotions.

[SITB-OSED](https://zenodo.org/records/16210020) (Indic-Odia Speech Emotion Data) is
an acted Odia corpus: 20 actors, 4 for each of five Odia dialects (Baleswari,
Berhampuri, Cuttacki, Phulbani, Sambalpuri), speak sentences in six emotions. The
archive has 12,110 wav files, 2.08 to 11.81 seconds long, 19.11 hours in total. The
recordings are 22.05 kHz (7 files 44.1 kHz) and mostly stereo (11,620 stereo, 483
mono), so `create_dataset` converts them to 16 kHz mono with ffmpeg.

The label is the second field of the file name `<prefix><sample>-<emotion>-<dialect>-<actor>.wav`
(for example `a58-01-02-07.wav`): `01` angry, `02` disgust, `03` fear, `04` happy,
`05` sad, `06` surprise. The builder also checks that the sample prefix (`a`, `d`,
`f`, `h`, `s`, `sr`) agrees with the code. One file, `sr102-06-10.wav`, has no
dialect field and is still read.

Environment variables (both optional; defaults are `download/` and `data/` under the
recipe):

- `SITB_OSED` — the directory holding the extracted `Actor_*` directories, or where to download them to
- `SITB_OSED_OUTPUT` — where the converted audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Zenodo record 16210020 (SITB-OSED, Indic-Odia Speech Emotion Data; `Actor_*` directories, downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Zenodo, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 12,110 | 20 | 19.11 |

How these were counted: every `.wav` under the `Actor_*` directories (mostly 22.05 kHz, 7 files at 44.1 kHz; 11,620 stereo and 483 mono; 2.08 to 11.81 s); hours from the file lengths. Speakers are the 20 actors (4 for each of five Odia dialects; odd actor numbers are women and even numbers men, per the description).

### Files per label

| emotion code in the file name | read as | samples |
|---|---|---|
| 01 | angry | 1,997 |
| 02 | disgust | 2,043 |
| 03 | fear | 1,990 |
| 04 | happy | 1,995 |
| 05 | sad | 2,118 |
| 06 | surprise | 1,967 |

How these were counted: the second field of the file name `<prefix><sample>-<emotion>-<dialect>-<actor>.wav` (for example `a58-01-02-07.wav`); the sample prefix (`a`, `d`, `f`, `h`, `s`, `sr`) agrees with the code. The release has no neutral class.

### Issues in the release

- One file, `sr102-06-10.wav`, has no dialect field.
- The description gives 3.5 to 8 s per file; the files are 2.08 to 11.81 s.
- The utterances are not transcribed in the release.


## Download

`prepare_source` downloads `SITB-OSED.zip` (4.9 GB, md5-checked) from
[Zenodo](https://zenodo.org/records/16210020) and extracts the wav files. The
download time was not measured. With the archive already extracted, `create_dataset`
took 11 minutes 1 second and `remove_long_short` 17 seconds.

## Splits

The archive publishes no partition. Whole actors are held out (`n_test_speakers`,
`n_valid_speakers`, `split_seed` in `dataset/config.yaml`): actors 11, 15, 17 and 19
for test, 1 and 18 for valid, the other 14 for training, so no split shares an
actor. Every actor reads the same prompts, so sentences are shared across splits.

| split | utterances | hours | after `remove_long_short` | angry | disgust | fear | happy | sad | surprise |
|---|---|---|---|---|---|---|---|---|---|
| train | 8,448 | 13.30 | 8,448 | 1,391 | 1,427 | 1,391 | 1,395 | 1,473 | 1,371 |
| valid | 1,279 | 2.13 | 1,279 | 209 | 207 | 209 | 206 | 241 | 207 |
| test | 2,383 | 3.68 | 2,383 (not filtered) | 397 | 409 | 390 | 394 | 404 | 389 |

## License

The Zenodo record states CC BY 4.0. A citation is not given in this recipe because
none was checked.

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
