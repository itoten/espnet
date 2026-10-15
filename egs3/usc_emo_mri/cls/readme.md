# USC-EMO-MRI speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on USC-EMO-MRI alone and scores it on its own test split, classifying each
utterance into one of four emotions.

[USC-EMO-MRI](https://zenodo.org/records/19325044) is an acted English corpus: five
male (`m1`-`m5`) and five female (`f1`-`f5`) actors read prompted sentences and
passages in four emotions, `neutral`, `angry`, `happy` and `sad`, at a normal and a
fast speaking rate. The speech was recorded inside an MRI scanner together with
real-time MRI video of the vocal tract, so the audio is noisy. The archive stores each
recording as an AVI file (20 kHz mono PCM audio plus a small video stream); this
recipe uses only the audio.

Utterances are cut out of the AVI files with the start and end times in
`labels/utt_time_stamp`. The archive lists 2,363 utterances; 29 have no `meta` row
and 29 carry emotion ID 5, which the README does not describe, so those 58 are
dropped and 2,305 remain (2.36 hours, 1.31 to 10.2 seconds each). The 3 AVI files
under `palate_sweep/` have no labels and are not used. One speaker's
`utt_time_stamp` has its first two columns swapped; the builder handles both orders.
`create_dataset` converts the audio to 16 kHz mono with ffmpeg.

The label is the `emotion ID` of `labels/meta`, which the README describes as the
final emotion label determined from the listener evaluation (at least ten listeners
per speaker according to the Zenodo description); the per-listener ratings are not in
the archive. `dataset/config.yaml` maps `1` neutral, `2` angry, `3` happy, `4` sad.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `USC_EMO_MRI` — the directory holding the extracted speaker directories (`USC-EMO-MRI/f1` ... `m5`), or where to download them to
- `USC_EMO_MRI_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `USC-EMO-MRI.zip` (1.49 GB, md5-checked) from
[Zenodo](https://zenodo.org/records/19325044) and extracts the AVI and label files
(2.1 GB). The download time was not measured; extracting took 13 seconds. With the
archive extracted, `create_dataset`, `remove_long_short` and `prepare_labels` together
took about 2 minutes 30 seconds.

## Splits

The archive publishes no partition. Whole speakers are held out (`n_test_speakers`,
`n_valid_speakers`, `split_seed` in `dataset/config.yaml`): `m3` and `m4` for test,
`f2` for valid and the other seven for training, so no split shares a speaker. All
speakers read the same prompts, so sentences are shared across splits. Valid is a
single speaker, so its score is noisy.

| split | utterances | hours | after `remove_long_short` | neutral | angry | happy | sad |
|---|---|---|---|---|---|---|---|
| train | 1,606 | 1.66 | 1,606 | 462 | 393 | 369 | 382 |
| valid | 218 | 0.22 | 218 | 62 | 52 | 52 | 52 |
| test | 481 | 0.48 | 481 (not filtered) | 132 | 118 | 121 | 110 |

## License

The Zenodo record states CC BY 4.0. The reference named in the archive README
(Kim et al., "Vocal tract shaping of emotional speech", J. Acoust. Soc. Am.) is
marked there as to be updated; its details were not checked, so no citation is given.

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
