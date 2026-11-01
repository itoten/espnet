# Telugu Emotion Speech speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Telugu Emotion Speech alone and scores it on its own test split,
classifying each utterance into one of five emotions.

[Telugu Emotion Speech](https://www.kaggle.com/datasets/jettysowmith/telugu-emotion-speech)
is a Kaggle upload with no description, so the recording conditions and the
labelling procedure are not documented. The release holds 458 wav files in
`telugu/<emotion>/`: `angry` 96, `happy` 94, `nuetral` 103, `sad` 80 and
`suprised` 85 (the folder names are misspelled in the release; the recipe maps them
to `neutral` and `surprise`). The audio is 44.1 kHz stereo (nine files are 48 kHz),
16-bit, 0.26 to 27.2 seconds long, 0.53 hours in total.

File names look like `S45_SRI_C01_G1_D02_SPKF23_V1_SA5_MMM.wav`. The `SPK[MF]<nn>`
field is read as the speaker; there are 20 such values (10 `SPKF`, 10 `SPKM`), but
what the other fields mean is not documented. The label is the folder name. Three
files in `nuetral/` carry the code `SA` in their names (the other 455 match their
folder); the recipe keeps the folder label. Two files in `sad/` are empty (0 bytes)
and are skipped, so 456 recordings are used.

Three environment variables control where the data lives. The last two are optional
and default to `download/` and `data/` under the recipe.

- `TELUGU_EMOTION_KAGGLE` — the directory holding the extracted `telugu/` folder
- `TELUGU_EMOTION_KAGGLE_OUTPUT` — where the converted audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Kaggle dataset `jettysowmith/telugu-emotion-speech` (downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Kaggle download with a free account and API token.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 458 | 20 | 0.53 |

How these were counted: every `.wav` under `telugu/<emotion>/` (44.1 kHz stereo, 16-bit; nine files are 48 kHz; 0.26 to 27.2 s); hours from the file lengths. Speakers are the 20 distinct `SPK[MF]nn` fields in the file names (10 `SPKF`, 10 `SPKM`); the release does not document the file name fields.

### Files per label

| label (folder name in the release) | samples |
|---|---|
| angry | 96 |
| happy | 94 |
| nuetral | 103 |
| sad | 80 |
| suprised | 85 |

How these were counted: the name of the folder each file is in. The folder names `nuetral` and `suprised` are spelled so in the release.

### Issues in the release

- The Kaggle page has no description, subtitle, tags or citation, so the recording conditions, the labelling procedure and the source are not documented.
- 2 files in the `sad` folder are empty (0 bytes).
- 3 files in the `nuetral` folder (speaker `SPKM03`) carry the code `SA` in their names; the other 455 files' codes match their folder.
- Observation by the recipe author (2026-10-18 at the latest): the 3 `nuetral` files with the `SA` code were listened to and sounded neutral. Three further `nuetral` files (`S95_BIL_CO14_G3`) were listened to and sounded angry; their F0 is close to that of the `angry` files. Only part of the files was listened to.
- Observation by the recipe author (2026-10-18 at the latest): the recordings sound like recorded speech, not like film audio.
- Whisper identified Telugu in 382 of 456 transcribed files; the rest include English lines.


## Download

The data is obtained manually; the recipe raises an error with the URL when it is
missing. Download the zip from
[Kaggle](https://www.kaggle.com/datasets/jettysowmith/telugu-emotion-speech) (a
free Kaggle account is needed), unzip it and set `$TELUGU_EMOTION_KAGGLE` to the
directory that holds `telugu/`. Measured on one run: about 10 seconds to download
the 0.2 GB zip with the Kaggle REST API; `create_dataset`, `remove_long_short` and
`prepare_labels` together took about 54 seconds (four recipes were prepared at the
same time).

## Splits

The release publishes no partition. Whole speakers are held out: 4 for test, 2 for
valid and 14 for train (`n_test_groups`, `n_valid_groups`, `split_seed` in
`dataset/config.yaml`). With `split_seed: 0` the four test speakers are all `SPKM`
(male), so test scores say nothing about female speakers.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 312 | 312 |
| valid | 47 | 47 |
| test | 97 | 97 (not filtered) |

`remove_long_short` drops nothing.

## License

The Kaggle page lists the license as Apache 2.0. No citation is given on the page.

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
