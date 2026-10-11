# BanglaSER speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on BanglaSER alone and scores it on BanglaSER's own test split, classifying each
utterance into one of five emotions.

[BanglaSER](https://data.mendeley.com/datasets/t9h6p943xy/5) (Das, Islam and
Ahmed; Data in Brief 42, 2022) is acted Bangla speech from 34 non-professional
speakers (17 male, 17 female, aged 19 to 47), each reading 3 statements 3 times
in `angry`, `happy`, `sad` and `surprise`, and 27 of them also in `neutral`.
There are 1,467 recordings (306 for each of the four emotions, 243 `neutral`),
1.37 hours in total, each 2.6 to 4.0 seconds long. There is no `disgust` or `fear`.

The label comes from the file name, in the RAVDESS style,
`<mode>-<type>-<emotion>-<intensity>-<statement>-<repetition>-<actor>.wav`
(e.g. `03-01-03-02-02-02-05.wav`); the emotion codes are 01 happy, 02 sad,
03 angry, 04 surprise and 05 neutral, as in the paper's Table 4. Two quirks in
the release are handled: the three surprise recordings of actor 15 have mode
`01` instead of `03`, and actor 19's folder holds a file named like actor 3's
(`03-01-05-01-01-03-03.wav`, with different audio). The speaker is therefore
taken from the `Actor NN` folder, and utterance ids include it. The recordings
are 44.1 or 48 kHz, mono or stereo; `create_dataset` converts them to 16 kHz mono
with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `BANGLASER` — the directory holding the extracted audio, or where to download it to
- `BANGLASER_OUTPUT` — where the resampled audio and manifests are written

## Download

`prepare_source` downloads the archive that Mendeley Data builds on request
(446 MB) and extracts the WAV files. The corpus is CC BY 4.0. Mendeley publishes no
checksum for it, so completeness is checked by counting the 1,467 files.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out, so the splits share all
34 speakers.

| split | utterances |
|---|---|
| train | 1026 |
| valid | 148 |
| test | 293 |

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
