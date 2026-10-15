# EESC speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on the Estonian Emotional Speech Corpus (EESC) alone and scores it on EESC's own
test split, classifying each utterance into one of four emotions.

[EESC](https://doi.org/10.6084/m9.figshare.25249744) (Institute of the Estonian
Language) is read-out, non-acted Estonian speech from one female speaker, with the
emotion of each sentence checked by perception tests (Altrov & Pajupuu, 2012). The
release holds 1164 sentences, 1.08 hours in total, 0.45 to 14.0 seconds long:
338 neutral, 306 anger, 271 joy and 249 sadness. (The figshare description says 1234
sentences; the archive holds 1164 `.wav` files, each with a `.TextGrid`.)

The label is the intended emotion written in the last tier (`emotion`) of each
`<id>.TextGrid`. `dataset/config.yaml` maps `anger`, `joy`, `sadness` and `neutral`
to `angry`, `happy`, `sad` and `neutral`. The recordings are 44.1 kHz 16-bit mono, so
`create_dataset` converts them to 16 kHz with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EESC` — the directory holding the extracted `*.wav` and `*.TextGrid` files, or where to download them to
- `EESC_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `ekorpus.zip` (277 MB, md5-checked) from
[figshare](https://doi.org/10.6084/m9.figshare.25249744) and extracts the `.wav` and
`.TextGrid` files. Measured on one run: `prepare_source` took about 3 minutes
(a plain `curl` of the zip alone took 93 seconds; another download was running at
the same time); `create_dataset`, `remove_long_short` and `prepare_labels`
together took about 85 seconds.

## Splits

The release publishes no partition and has a single speaker. Utterances are split
70/10/20 at random within each label (`train_ratio`, `valid_ratio`, `test_ratio`,
`split_seed` in `dataset/config.yaml`). The splits share the speaker, and the
1164 recordings hold only 1083 distinct sentence texts (80 texts occur more than once),
so the same sentence can fall in different splits and scores are optimistic.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 814 | 812 |
| valid | 117 | 117 |
| test | 233 | 233 (not filtered) |

`remove_long_short` drops the 2 train clips shorter than 0.5 s.

## License and citation

The figshare record is licensed CC BY 4.0. It cites: Altrov, R. and Pajupuu, H.
2012. Estonian Emotional Speech Corpus: Theoretical base and implementation. In:
4th International Workshop on Corpora for Research on Emotion Sentiment & Social
Signals (ES3), 50-53. Istanbul.

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
