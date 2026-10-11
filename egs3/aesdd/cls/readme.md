# AESDD speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on AESDD alone and scores it on AESDD's own test split, classifying each
utterance into one of five emotions.

[AESDD](https://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/)
(Acted Emotional Speech Dynamic Database v1.0, Aristotle University of
Thessaloniki, 2017) is acted Greek speech from actors (the original release used professional actors), six
speakers in the files this recipe downloads. Each speaker acts utterances of
ambiguous, out-of-context content in five emotions: 604 recordings with audio,
`angry`, `disgust`, `fear`, `happy` and `sad` (119 to 122 each), 0.69 hours in
total, each 1.0 to 12.7 seconds long. There is no `neutral` or `surprise`.

The emotion comes from the folder (`anger`, `disgust`, `fear`, `happiness`,
`sadness`); the file name `<code><utterance> (<speaker>)[<take>].wav` (e.g.
`a03 (4).wav`) starts with the same one-letter code, and `create_dataset` fails
if the two disagree. One recording, `sadness/s05 (3).wav`, is a WAV header with
no samples, so it is skipped. The sources are 44.1 kHz mono (float32 or 24-bit
PCM), so `create_dataset` resamples them to 16 kHz 16-bit with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `AESDD` — the downloaded corpus, or where to download it to
- `AESDD_OUTPUT` — where the resampled audio and manifests are written

## Download

`prepare_source` downloads the corpus from the public MEGA folder linked on the
corpus's own page (about 410 MB, with its documentation). MEGA encrypts files
client-side, so the builder decrypts them itself with the folder key from that
link. The page offers the corpus for non-commercial research use.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out.

| split | utterances |
|---|---|
| train | 424 |
| valid | 60 |
| test | 120 |

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
