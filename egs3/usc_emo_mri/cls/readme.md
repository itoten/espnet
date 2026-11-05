# USC-EMO-MRI speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on USC-EMO-MRI alone and scores it on its own test split, classifying each
utterance into one of four emotions.

[USC-EMO-MRI](https://zenodo.org/records/19325044) is an acted English corpus: five
male (`m1`-`m5`) and five female (`f1`-`f5`) actors, all with professional acting
experience, read prompted sentences and passages in four target emotions, `neutral`,
`angry`, `happy` and `sad`, at a normal speaking rate (`neutral` also at a fast rate,
98 utterances). The speech was recorded inside an MRI scanner together with real-time
MRI video of the vocal tract; the paper states that noise cancellation was applied to
the audio afterwards. The archive stores each recording as an AVI file (20 kHz mono
PCM audio plus a small video stream); this recipe uses only the audio.

Utterances are cut out of the AVI files with the start and end times in
`labels/utt_time_stamp`. The archive lists 2,363 utterances; 29 have no `meta` row
and 29 carry emotion ID 5, which the README does not describe, so those 58 are
dropped and 2,305 remain (2.36 hours, 1.31 to 10.2 seconds each).
26 of the 356 AVI files (including the 3 under `palate_sweep/`) appear in no
`utt_time_stamp` and are not used. One speaker's
`utt_time_stamp` has its first two columns swapped; the builder handles both orders.
`create_dataset` converts the audio to 16 kHz mono with ffmpeg and then reduces its
reverberation (next section).

The label is the `emotion ID` of `labels/meta`, which the README describes as the
final emotion label determined from the listener evaluation. According to the paper,
10 to 12 listeners per speaker chose the perceived emotion and the final label is the
majority vote, so it is the emotion the listeners perceived, not the one the actor was
asked to portray; the per-listener ratings are not in the archive. `dataset/config.yaml` maps `1` neutral, `2` angry, `3` happy, `4` sad.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `USC_EMO_MRI` — the directory holding the extracted speaker directories (`USC-EMO-MRI/f1` ... `m5`), or where to download them to
- `USC_EMO_MRI_OUTPUT` — where the converted audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Zenodo record 19325044 (archive `USC-EMO-MRI`, 10 speaker directories, downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Zenodo, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 2,363 | 10 | 2.42 |

How these were counted: every row of the `labels/utt_time_stamp` files of the 10 speakers (one utterance per row); hours from the start and end times in those rows (the audio is stored inside 356 `.avi` files, 20 kHz mono PCM, together with a vocal-tract video). Speakers are the 10 speaker directories (`m1` to `m5`, `f1` to `f5`).

### Files per label

| emotion ID in `labels/meta` | meaning (README) | samples | hours |
|---|---|---|---|
| 1 | neutral | 656 | 0.650 |
| 2 | angry | 563 | 0.559 |
| 3 | happy | 542 | 0.544 |
| 4 | sad | 544 | 0.610 |
| 5 | not described | 29 | 0.031 |
| (no row in `meta`) | - | 29 | 0.024 |

How these were counted: the `emotion ID` column of the `labels/meta` files of the 10 speakers (2,334 rows); 29 of the 2,363 time-stamp rows have no row in `meta`. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- Emotion ID 5 (29 utterances) is not described in the README; the paper's evaluation offers an "other" option, but no document says that ID 5 means "other".
- 29 utterances have a time-stamp row but no `meta` row; the reason is not given in the documentation.
- 26 of the 356 `.avi` files (3 of them under `palate_sweep/`) appear in no `utt_time_stamp` file; their content was not examined.
- One speaker's `utt_time_stamp` has its first two columns swapped (utterance ID and AVI basename).
- The per-utterance ratings of the listeners are not in the archive; the labels are the final labels only.
- Observation by the recipe author (2026-10-19 at the latest): some utterances were listened to; they sound reverberant, like a reverb effect. The recordings were made inside an MRI scanner; the cause of the sound was not checked.


## Reverberation reduction (provisional)

Even after the authors' noise cancellation, every recording sounds reverberant,
probably because it was made inside the MRI scanner. `create_dataset` therefore
passes each utterance through the pretrained ESPnet enhancement model
[`espnet/Wangyou_Zhang_universal_train_enh_uses_refch0_2mem_raw`](https://huggingface.co/espnet/Wangyou_Zhang_universal_train_enh_uses_refch0_2mem_raw)
(USES; its model card lists REVERB and WHAMR among the training data) and mixes the output back
with the input: `0.9 * enhanced + 0.1 * input`, with the enhanced signal scaled to
the input level. The model output alone sounded choppy, and the input share makes
it less so. The model is downloaded from Hugging Face on the first run and uses a
GPU when one is available.

This processing is provisional and not optimized. The model and the mixing weight
(`enhancement` in `dataset/config.yaml`) were chosen by listening to six training
utterances against a second model and a few other weights; they were not tuned for
recognition accuracy, and the effect on scores has not been measured.

The processing can be switched off with `enhancement.enabled: false` in
`dataset/config.yaml`. The setting is only read when the audio is written, so after
changing it, delete `wav/` and the manifests in `$USC_EMO_MRI_OUTPUT` before running
`create_dataset` again.

## Download

`prepare_source` downloads `USC-EMO-MRI.zip` (1.49 GB, md5-checked) from
[Zenodo](https://zenodo.org/records/19325044) and extracts the AVI and label files
(2.1 GB). The download time was not measured; extracting took 13 seconds. With the
archive extracted, `create_dataset` (cutting, resampling and the reverberation reduction
on one GPU) took 8 minutes 28 seconds, `remove_long_short` 14 seconds and
`collect_stats` 19 seconds.

## Splits

The archive publishes no partition. Whole speakers are held out (`n_test_speakers`,
`n_valid_speakers`, `split_seed` in `dataset/config.yaml`): `m3` and `m4` for test,
`f2` for valid and the other seven for training, so no split shares a speaker. Sentences
are shared across splits: all speakers read sentences 1-6 and the passage (9-18),
while sentence 7 (reiterant "Ma Ma Ma ...") was recorded for six speakers only. Valid is a
single speaker, so its score is noisy.

| split | utterances | hours | after `remove_long_short` | neutral | angry | happy | sad |
|---|---|---|---|---|---|---|---|
| train | 1,606 | 1.66 | 1,606 | 462 | 393 | 369 | 382 |
| valid | 218 | 0.22 | 218 | 62 | 52 | 52 | 52 |
| test | 481 | 0.48 | 481 (not filtered) | 132 | 118 | 121 | 110 |

## License and citation

The Zenodo record states CC BY 4.0, which requires attribution. Zenodo gives this
citation:

Kim, J., Toutios, A., Kim, Y.-C., Zhu, Y., Lee, S., & Narayanan, S. (2014). USC-EMO-MRI:
An emotional speech production database [Dataset]. 10th International Seminar on Speech
Production (ISSP). USC-EMO-MRI corpus: An emotional speech production database recorded
by real-time magnetic resonance imaging, Cologne, Germany.
https://doi.org/10.5281/zenodo.19325044

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
