# KBES speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on KBES alone and scores it on its own test split, classifying each recording
into one of five emotions.

[KBES](https://data.mendeley.com/datasets/vsn37ps3rx/4) (KUET Bangla Emotional
Speech, Mendeley Data, V4, doi:10.17632/vsn37ps3rx.4) is Bangla speech cut from
publicly available telefilms, dramas, TV series and web series on YouTube and
Facebook. Billah, Sarker and Akhand describe it in "KBES: A dataset for realistic
Bangla speech emotion recognition with intensity level", Data in Brief 51 (2023)
109741, doi:10.1016/j.dib.2023.109741 (the paper says there are 35 actors, all
anonymized). The labels are `neutral`, `happy`, `sad`, `angry` and `disgust`. There
is no `fear` or `surprise`.

## What is used

The release has 900 clips of about 3 seconds each (all 48 kHz stereo 16-bit PCM,
3.02 to 3.16 seconds). The file names are `<emotion>-<intensity>-<gender>-<number>.wav`
(for example `2-1-0-001.wav`); the paper's Table 4 gives the codes: emotion 1
neutral, 2 happy, 3 sad, 4 angry, 5 disgust; intensity 1 low or 2 high (the neutral
clips are all intensity 1); gender 0 female or 1 male. The label is the emotion
code; the intensity and gender are not used. Each of the 18 combinations has 50
clips (neutral has 2 × 50, one per gender).

**20 pairs of clips have identical content** (SHA-256): 17 in happy-high, 1 in
angry-low and 2 in angry-high, all within the same label. Of each pair only the
first by file name is kept, so 880 recordings (0.738 hours) are used. This keeps a
clip from appearing in both train and test.

`create_dataset` converts the clips to 16 kHz mono 16-bit WAV with ffmpeg;
`remove_long_short` drops nothing.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `KBES` — the directory holding `audio/`, or where to download it to
- `KBES_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` reads the Mendeley Data API listing, downloads the 900 WAV files
(500 MB) with 8 threads, and checks each one's size and SHA-256 against the listing.
Mendeley rejects requests that do not send a browser-like `User-Agent`, so the
recipe sends one. The dataset is licensed CC BY 4.0 (the license field of the
Mendeley listing). Please cite the paper above. The clips come from third-party
videos; the paper states that the authors checked there are no copyright issues.

## Splits

The release publishes no partition, and the file names do not say who speaks.
Recordings are split 70/10/20 at random within each label (`train_ratio`,
`valid_ratio`, `test_ratio`, `split_seed` in `dataset/config.yaml`), so the splits
share actors.

| split | recordings | after `remove_long_short` | angry | disgust | happy | neutral | sad |
|---|---|---|---|---|---|---|---|
| train | 616 | 616 | 138 | 140 | 128 | 70 | 140 |
| valid | 88 | 88 | 20 | 20 | 18 | 10 | 20 |
| test | 176 | (not filtered) | 39 | 40 | 37 | 20 | 40 |

`neutral` has half as many clips as the other labels, so its class weight is 1.76
(the others 0.88 to 0.96). Report Macro-F1 rather than accuracy.

## Time needed

Measured on this machine (CPU only, `CUDA_VISIBLE_DEVICES=""`). The download ran
while two other Mendeley downloads were running, so it depends on the network.

| step | time |
|---|---|
| `prepare_source` (download of 900 files, 500 MB, with size and SHA-256 checks) | 299.5 s |
| `create_dataset` (conversion to 16 kHz mono, one file at a time) | 55.2 s |
| `remove_long_short` | 17.6 s |
| the whole `create_dataset remove_long_short prepare_labels` command | 80.8 s (includes starting `pixi` and Python) |

The downloaded audio takes 500 MB and the converted data 83 MB.

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
