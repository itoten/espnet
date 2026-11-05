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

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Hugging Face dataset `issai/KazEmoTTS` (repository revision `faae8a52c4df036e086dea50b5810677d296c017`, last modified 2025-02-11; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Hugging Face, direct download, no sign-in or form. The project's own page asks for a Google form that returns FTP credentials; the Hugging Face repository is public and the form was not used.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 54,760 | 3 | 74.9 |

How these were counted: every `.wav` under `<narrator>/<train|eval>/` (22.05 kHz mono 16-bit, 0.65 to 18.16 s); hours from the file lengths. Speakers are the 3 narrator IDs (`1263201035` 19,802 files, `399172782` 10,302, `805570882` 24,656; one woman and two men, professional narrators). The release has an official train / eval split per narrator.

### Files per label

| label (emotion in the file name) | samples | hours |
|---|---|---|
| neutral | 9,385 | 12.694 |
| surprise | 9,179 | 12.717 |
| fear | 9,098 | 12.127 |
| angry | 9,059 | 12.021 |
| happy | 9,059 | 12.426 |
| sad | 8,980 | 12.874 |

How these were counted: the emotion field of each file name `<narrator>_<emotion>_<id>.wav`. The release has no disgust class. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- No byte-identical, decoded-identical or near-identical recordings were found.


## Download

`prepare_source` downloads `EmoKaz.zip` (10.5 GB) from Hugging Face, checks its
sha256 against the hash on the Hub, and extracts only the WAV files. The
corpus is CC BY 4.0 (stated on the authors' download form; the Hugging Face card itself
has no license field). The authors' form, linked from the card, offers the data by
FTP after registration; the Hugging Face repository is public and needs no login, which
is what this recipe uses. The form asks research users to cite the dataset and
commercial users to state "This product/service utilises the ISSAI KazEmoTTS dataset."

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
