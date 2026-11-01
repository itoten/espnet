# MDER-MA speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on MDER-MA alone and scores it on its own test split, classifying each
recording into one of four emotions.

[MDER-MA](https://data.mendeley.com/datasets/yzsw3ff6rn/1) (Ouali and El Garouani,
*Data in Brief* 62, 2025, doi:10.1016/j.dib.2025.112005) is Moroccan Arabic
(Darija) speech labeled `angry`, `happy`, `neutral` or `sad` by five native
speakers, according to the dataset description. There is no `disgust`, `fear` or
`surprise`. The dataset also holds transcripts and spectrogram images, which this
recipe does not use.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Mendeley Data record `yzsw3ff6rn` version 1 (MDER-MA; downloaded around 2026-10-15). The release holds 1,345 wav files; the local copy kept after removing unusable and repeated files has 1,267, and the counts below are those of the 1,267 (the 78 removed files are described under the issues). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Mendeley Data, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 1,267 (of 1,345 in the release) | unknown (66 gender-age-ID combinations) | 2.11 |

How these were counted: the `.wav` files of the local copy (48 kHz stereo 16-bit; 1 to 51 s, all whole seconds, median 6 s); hours from the file lengths. The file names carry a gender, an age and a numeric user ID that the paper says allows following one speaker (66 distinct combinations, ages 13 to 68, 232 recordings under 18), but the names are irregular, so the number of speakers is unknown.

### Files per label

| label (first word of the file name, case-folded) | samples (of the 1,267) |
|---|---|
| sad | 384 |
| neutral | 312 |
| happy | 303 |
| angry | 268 |

How these were counted: the first word of each file name (`Angry_male_44_1_9-xjWLfu.wav`, `sad_chunk_3.wav`, `sad0.wav`). The release has no disgust, fear or surprise class. According to the paper five native speakers listened to every segment and kept those on which at least three agreed (Fleiss' kappa 0.89).

### Issues in the release

- Of the 1,345 files of the release, 45 are 44-byte headers without audio and 33 repeat the content (SHA-256) of another file with the same label; these 78 are not in the 1,267. 17 file names are used for two different recordings (they are kept apart by adding the start of the SHA-256).
- The paper states 1,322 samples, 16 kHz mono and about 3 to 20 s; the files are 1,345 in number, 48 kHz stereo and 1 to 51 s. The cause was not investigated.
- File names are irregular: 29 `sad` files (`sad_chunk_N`, `sadN`) have no gender, age or ID; 21 files have the typo `fmale`; some files have a random suffix. In 176 recordings the correlation between the left and right channel is 0.9 or lower.


## What is used

The 1,345 WAV files of the Mendeley Data release are not all usable, so the recipe
keeps 1,267 of them (2.11 hours):

- 45 files are 44-byte headers without any audio, and are left out.
- 33 of the rest repeat the content (same SHA-256) of another file, and only the first
  by name is kept. The repeats carry the same label.
- 17 file names are used for two different recordings, so a file is stored as
  `<name>__<first 12 hex digits of its SHA-256>.wav`.

The label is the first word of the file name (`Angry_male_44_1_9-xjWLfu.wav`,
`sad_chunk_3.wav`, `sad0.wav`). Names are otherwise irregular (a `fmale` typo, a
random suffix on some files, `sad` files without gender, age or session). The
paper says the file name holds a gender, an age, a numeric `UserID` assigned to each
speaker ("allowing for tracking of multiple samples from the same individual") and a
sample number; the irregular names are why speakers are not parsed here. The paper says 1,322 samples per modality; the
file counts in the release do not match that, and the cause was not investigated.
The source audio is 48 kHz stereo 16-bit; `create_dataset` converts it to 16 kHz
mono. Durations range from 1 to 51 seconds (median 6; all are whole seconds), so
`remove_long_short` drops 5 recordings (4 from train longer than 30 seconds, and
1 of exactly 30.0 seconds from valid).

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `MDERMA` — the directory holding `audio/*.wav`, or where to download it to
- `MDERMA_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` reads the file listing of the dataset from the Mendeley Data API
and downloads the 1,267 recordings with 8 threads. The listing gives each file's
size and SHA-256, and both are checked. Mendeley rejects requests that do not send
a browser-like `User-Agent`, so the recipe sends one. The dataset is the authors'
own upload and is licensed CC BY 4.0 (the license field of the Mendeley listing).
Please cite the paper above.

**Source rights.** The audio is cut from Moroccan reality TV shows, radio programs,
podcasts and interviews found online. The paper (Data in Brief 62 (2025) 112005, CC BY)
says in its Ethics Statement: "All data used in this study were derived from publicly
available, non-identifiable sources, and no personally identifiable information is
present." and "Authors have considered and followed the source websites' ToS,
privacy laws, and user consents. Copyright: The data does NOT belong to users (e.g.,
social media). The data are published for free through user consent, and they can be
consumed by anyone who has access to the Internet, so ethical approval has not been
sought." It also says "We are not using the data for any fraudulent activities such as
making a profit (e.g., Business)", and in the collection section "Since all data were
obtained from publicly available online sources, no explicit consent was required for
collection and use." These are the authors' assertions. The paper names no source
program or site and gives no terms of service, so they cannot be checked, and it says
nothing about non-commercial use. Whether the clips are open data, and whether
non-commercial terms apply, is unconfirmed.

## Splits

The release publishes no partition. Recordings are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). The names carry a gender, an age and a `UserID` that the
paper says identifies a speaker, but they are irregular (see above), so speakers
are not held out and the splits probably share speakers.

| split | recordings | after `remove_long_short` | angry | happy | neutral | sad |
|---|---|---|---|---|---|---|
| train | 887 | 883 | 187 | 212 | 219 | 269 |
| valid | 126 | 125 | 27 | 30 | 31 | 38 |
| test | 254 | (not filtered) | 54 | 61 | 62 | 77 |

## Time needed

Measured on this machine (CPU only, `CUDA_VISIBLE_DEVICES=""`). The download
depends on the network (Mendeley was slow here).

| step | time |
|---|---|
| `prepare_source` (1,267 files, 1.4 GB, 8 threads, SHA-256 checked) | 8 min 45 s |
| `create_dataset` (conversion to 16 kHz, one file at a time) | 81.6 s |
| `remove_long_short` | 17.0 s |
| the whole `create_dataset remove_long_short prepare_labels` command | 1 min 47 s (includes starting `pixi` and Python) |

The downloaded audio takes 1.4 GB and the converted data 235 MB.

## Listening test

Five native Moroccan Arabic speakers listened to every segment separately, and only segments on which at least three agreed were kept (Fleiss' kappa 0.89, per the paper). The individual judgements are not distributed.

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
