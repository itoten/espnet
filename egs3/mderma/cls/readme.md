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
random suffix on some files, `sad` files without gender, age or session), which
is why speakers are not parsed. The paper says 1,322 samples per modality; the
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

## Splits

The release publishes no partition. Recordings are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). The names carry a gender, an age and a session number,
but the release does not say they identify a speaker, so speakers are not held
out and the splits probably share speakers.

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
