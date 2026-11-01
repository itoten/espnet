# Kannada speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Kannada alone and scores it on Kannada's own test split, classifying each
utterance into one of six emotions.

[A Kannada Emotional Speech Dataset](https://zenodo.org/records/6345107)
(Vishakha Agrawal, 2022) is Kannada read speech from 13 volunteers (9 female,
4 male, aged 12 to 48), each reading the same six sentences in six emotions:
468 utterances, 78 per emotion (`angry`, `fear`, `happy`, `neutral`, `sad`,
`surprise`; no `disgust`), 0.50 hours in total, each 1.5 to 7.2 seconds long.
The recordings were made by the volunteers themselves, not in a controlled
environment.

The label comes from the file name, `<speaker>-<emotion>-<sentence>.wav` (e.g.
`09-03-02.wav` is speaker 09, emotion 03, sentence 02); `dataset/config.yaml`
maps the six emotion numbers to labels. The sampling rate and channel count
differ between files (16, 44.1 and 48 kHz; mono and stereo), so `create_dataset`
converts every recording to 16 kHz mono with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `KANNADA` — the directory of WAV files, or where to download them to
- `KANNADA_OUTPUT` — where the converted audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Zenodo record 6345107 (A Kannada Emotional Speech Dataset; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Zenodo, no sign-in. The download often fails with `ContentTooShort`; the files were fetched one by one and checked by MD5.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 468 | 13 | 0.50 |

How these were counted: every `.wav` in the archive (1.50 to 7.20 s; mono at 44.1 kHz in 339 files, 48 kHz in 80, 16 kHz in 25, and stereo at 48 kHz in 17 and at 44.1 kHz in 7); hours from the file lengths. Speakers are the 13 speaker numbers in the file names `<speaker>-<emotion>-<sentence>.wav` (volunteers, 4 men and 9 women aged 12 to 48).

### Files per label

| emotion number in the file name | read as (recipe mapping) | samples |
|---|---|---|
| 01 | angry | 78 |
| 02 | sad | 78 |
| 03 | surprise | 78 |
| 04 | happy | 78 |
| 05 | fear | 78 |
| 06 | neutral | 78 |

How these were counted: the second field of each file name (13 speakers x 6 emotions x 6 sentences). The release has no disgust class.

### Issues in the release

- The recordings were made by the volunteers themselves, not in a controlled environment; the equipment and format differ between speakers.
- The 17 stereo files of speaker 05 come from two microphones whose channels are offset by about 18 to 22 samples; averaging them to mono dents the spectrum around 0.9 to 1.5 kHz by 4.9 to 8.5 dB (other bands 0.7 to 4.8 dB).
- Three pairs of files have the same sound: in two pairs the emotions differ (`09-03-02` surprise and `09-04-02` happy; `13-04-04` happy and `13-06-04` neutral) and in one pair both are neutral (`09-06-04` and `09-06-05`).


## Download

`prepare_source` downloads the 468 WAV files one by one from
[Zenodo](https://zenodo.org/records/6345107) (the record has no archive),
checking each file's MD5 against the record and resuming after an interrupted
run. The corpus is CC BY 4.0.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out.

| split | utterances |
|---|---|
| train | 322 |
| valid | 48 |
| test | 93 |

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

**Duplicates.** Three pairs of recordings have the same sound. In two pairs the files carry different emotions (`09-03-02` surprise and `09-04-02` happy; `13-04-04` happy and `13-06-04` neutral); none of these four is used (`conflicting_ids`). In the third, `09-06-04` and `09-06-05` (both `neutral`, different sentence numbers), `09-06-05` is dropped with `exclude_duplicates: true` (the default); `false` keeps both. This leaves 463 recordings (464 with `false`).
