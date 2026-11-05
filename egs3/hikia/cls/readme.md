# HiKIA speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on HiKIA alone and scores it on HiKIA's own test split, classifying each
utterance into one of four emotions.

[Hi, KIA](https://zenodo.org/records/7091465) (Kim et al., APSIPA 2022) is a
set of Korean-accented wake-up word utterances ("Hi, KIA") from 8 speakers (4
female, 4 male) in four emotions, `angry`, `happy`, `neutral` and `sad`: 488
utterances (107, 129, 119 and 133), 0.09 hours in total, each only 0.24 to
1.75 seconds long.

The label comes from the last letter of the file name,
`<speaker>_S<scene>_<trial>_<emotion>.wav` (e.g. `F5_S07_1_h.wav` is happy, from
speaker F5); `dataset/config.yaml` maps `a`, `h`, `n` and `s` to labels. The
letters agree with the `emo` column of the release's `annotation.csv` for all
488 files. The recordings are 48 kHz stereo, so `create_dataset` converts them
to 16 kHz mono with ffmpeg.

The label is the emotion the voice actors were asked to express (the target emotion), not what listeners heard. The release also has a perceived-emotion annotation by eight listeners for every recording (see "Listening test"); it is not used as the label here.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `HIKIA` — the extracted `hi_kia/wav/` directory, or where to download it to
- `HIKIA_OUTPUT` — where the converted audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Zenodo record 7091465 (Hi, KIA; `hi_kia/wav` and `annotation/`; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Zenodo, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 488 | 8 | 0.087 |

How these were counted: every `.wav` in `hi_kia/wav` (48 kHz stereo 16-bit, 0.24 to 1.75 s); hours from the file lengths. Speakers are the 8 speaker codes at the start of the file names `<speaker>_S<scene>_<trial>_<emotion>.wav` (four female, four male). The release holds the 488 recordings that remained after 88 of 576 were removed by the authors (all eight raters chose another emotion).

### Files per label

| emotion letter in the file name | read as | samples | hours |
|---|---|---|---|
| a | angry | 107 | 0.018 |
| h | happy | 129 | 0.021 |
| n | neutral | 119 | 0.019 |
| s | sad | 133 | 0.029 |

How these were counted: the last letter of each file name; it agrees with the `emo` column of the release's `annotation.csv` for all 488 files. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- The recordings are very short (0.24 to 1.75 s): the wake-up phrase "Hi, KIA".
- In `annotation/human.csv` the letter codes differ from those of the file names (a = angry, e = happy, d = sad, n = neutral, u = unknown).
- The files `F5_S03_1_a` and `F5_S03_4_a` (both angry) have the same sound.

### Listening test ratings in the release

`annotation/human.csv` holds the votes of eight graduate students for every recording, each choosing one of the four emotions or unknown (rated in random order). The release was thinned to the 488 recordings for which at least one rater chose the intended emotion. Counted on the file: 63.2% of the votes equal the intended emotion (sad 81.6%, happy 62.7%, neutral 54.6%, angry 50.4%); the majority equals the intended emotion for 330 recordings, another emotion has the majority for 121 and the vote is tied for 37.


## Download

`prepare_source` downloads `hi_kia_1.0.tar.gz` (750 MB) from
[Zenodo](https://zenodo.org/records/7091465) and extracts only its `wav/`
directory; the rest of the archive is precomputed features and pretrained
weights. The corpus is CC BY 4.0.

## Splits

The release publishes eight speaker-held-out folds (`split/`), not a single
partition. This recipe does not use them: utterances are split 70/10/20 at
random within each label (`train_ratio`, `valid_ratio`, `test_ratio`,
`split_seed` in `dataset/config.yaml`), so all three splits contain all 8
speakers.

| split | utterances |
|---|---|
| train | 340 |
| valid | 49 |
| test | 98 |

The utterances are 0.24 s or longer, so `remove_long_short` drops none for being
short.

## Listening test

Eight graduate students rated 576 recordings in random order, choosing one of the four emotions or `unknown`; the 88 recordings for which all eight chose another emotion were removed, leaving the 488 here. The raters' accuracy is 63.17% (63.2% of the votes agree with the intended emotion; sad 81.6%, happy 62.7%, neutral 54.6%, angry 50.4%). `annotation/human.csv` in the release holds the eight votes of every recording (letters `a` angry, `e` happy, `d` sad, `n` neutral, `u` unknown, which differ from the file-name letters); the majority equals the intended emotion for 330 clips. The votes are not used here.

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

**Duplicates.** `F5_S03_1_a` and `F5_S03_4_a` (both `angry`) are the same recording (byte-identical). With `exclude_duplicates: true` (the default) `F5_S03_4_a` is dropped (`duplicate_ids`), leaving 487 recordings; `false` keeps both.
