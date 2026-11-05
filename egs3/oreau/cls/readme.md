# Oréau speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Oréau alone and scores it on Oréau's own test split, classifying each
utterance into one of seven emotions.

[Oréau](https://zenodo.org/records/4405783) is French read speech from 32
non-professional speakers, each emotion with its own sentences about everyday
classroom life (79 sentences in all). This recipe uses version 2 of the
release: the 434 of version 1's 502 utterances that listeners in a perception
test recognised and judged natural. Seven emotions: `neutral` (70), `angry`
(67), `disgust` (67), `happy` (62), `surprise` (62), `sad` (55) and `fear`
(51).

The label comes from the file name, `<speaker><sentence><emotion><suffix>.wav`
(e.g. `25a09Da.wav`); no annotation file is read. Because each emotion has its
own sentences, the words carry information about the label.

The recordings are 44.1 kHz stereo from two microphones. The two channels are
not identical in any file and are poorly correlated in some, so averaging them
can attenuate the speech; `create_dataset` keeps one channel (`channel` in
`dataset/config.yaml`) and resamples it to 16 kHz with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `OREAU` — a directory holding the extracted `OréauFR_02/`, or where to
  download it to
- `OREAU_OUTPUT` — where the resampled audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Zenodo record 4405783, version 2 (Oréau French emotional speech; `OréauFR_02`, a RAR archive; version 1 has 502 utterances; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Zenodo, no sign-in. The archive is a RAR file and needs `bsdtar` to extract.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 434 | 32 | 0.34 |

How these were counted: every `.wav` in `OréauFR_02` (44.1 kHz stereo 16-bit from two microphones, 1.05 to 6.78 s); hours from the file lengths. Speakers are the 32 speaker numbers in the file names `<speaker><sentence><emotion><suffix>.wav` (non-professional; the table in the documentation gives 8 women and 24 men aged 22 to 27, though the table also lists ages 28 and 42).

### Files per label

| emotion letter in the file name | read as (French) | samples | hours |
|---|---|---|---|
| N | neutral (neutre) | 70 | 0.044 |
| C | angry (colère) | 67 | 0.053 |
| D | disgust (dégoût) | 67 | 0.064 |
| J | happy (joie) | 62 | 0.042 |
| S | surprise | 62 | 0.047 |
| T | sad (tristesse) | 55 | 0.045 |
| P | fear (peur) | 51 | 0.041 |

How these were counted: the emotion letter in each file name. According to the documentation, version 2 keeps the utterances that at least 50% of the listeners in the perception test recognised correctly and found natural; the individual results are not distributed. Each emotion has its own sentences (79 in all). Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- The two recordings `m/sessc/30a07Ca.wav` (speaker 30, sentence 07, angry) and `m/sesst/41a11Ta.wav` (speaker 41, sentence 11, sad) have identical audio samples in both channels; only the headers differ.
- The documentation says the files were recorded at 44.1 kHz and reduced to 16 kHz; the files are 44.1 kHz stereo. The two channels (two microphones) are not identical in any file and are poorly correlated in some.


## Download

`prepare_source` downloads `OréauFR_02.rar` (109 MB) from
[Zenodo](https://zenodo.org/records/4405783). The corpus is CC BY 4.0.

The archive is RAR, which Python cannot read, so extracting it needs `bsdtar`
on `PATH` (from libarchive, e.g. the conda-forge `libarchive` package).

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out.

| split | utterances |
|---|---|
| train | 305 |
| valid | 44 |
| test | 85 |

## Listening test

Version 2 keeps the recordings that a perception test showed were recognised by more than 50% of the listeners and judged natural (434 of the 502 of version 1). The individual results are not distributed.

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

**Duplicates.** `m/sessc/30a07Ca.wav` (speaker 30, sentence 07, `angry`) and
`m/sesst/41a11Ta.wav` (speaker 41, sentence 11, `sad`) have identical samples in both
channels and differ only in the header, but carry different labels. Both are used by
default. `exclude_duplicates: true` in `dataset/config.yaml` drops `41a11Ta`, which looks
like the copy (`duplicate_ids`), leaving 433; the default is `false`.
