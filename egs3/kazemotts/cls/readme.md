# KazEmoTTS speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on KazEmoTTS alone and scores it on KazEmoTTS's own test split, classifying each
utterance into one of six emotions.

[KazEmoTTS](https://huggingface.co/datasets/issai/KazEmoTTS) (Abilbekov et al.,
arXiv:2404.01033) is acted Kazakh speech from 3 professional narrators (1 female,
2 male) in `angry`, `fear`, `happy`, `neutral`, `sad` and `surprise`. There are
54,760 recordings, 74.9 hours in total, each 0.6 to 18.2 seconds long. There is
no `disgust`.

The label comes from the file name, `<narrator>_<emotion>_<id>.wav` (e.g.
`399172782_fear_23989.wav`), and each file sits in `<narrator>/<train|eval>/`.
`create_dataset` fails if the narrator folder disagrees with the file name. The
recordings are 22.05 kHz mono 16-bit; `create_dataset` resamples them to 16 kHz
with ffmpeg, `num_workers` files at a time.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `KAZEMOTTS` — the directory holding the extracted audio, or where to download it to
- `KAZEMOTTS_OUTPUT` — where the resampled audio and manifests are written

## Download

`prepare_source` downloads `EmoKaz.zip` (10.5 GB) from Hugging Face, checks its
sha256 against the hash on the Hub, and extracts only the WAV files. The
corpus is CC BY 4.0.

## Splits

The `train` and `eval` folders are the corpus's own split, and `eval` is used as
the test set. The validation set is a random 10% of the `train` recordings
within each label (`valid_ratio`, `split_seed` in `dataset/config.yaml`).
Narrators are not held out, so all three splits share the 3 narrators.

| split | utterances |
|---|---|
| train | 45928 |
| valid | 5102 |
| test | 3730 |

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
