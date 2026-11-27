# Emo-Emilia speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Emo-Emilia alone and scores it on a held-out part of Emo-Emilia,
classifying each clip into one of seven emotions.

[Emo-Emilia](https://huggingface.co/datasets/ASLP-lab/Emo-Emilia) was released
with the C<sup>2</sup>SER paper as a speech emotion recognition test set. Its
clips are a subset of Emilia, a large multilingual speech-generation corpus of
in-the-wild speech. The set has 1,400 clips, 100 per emotion in each of two
languages (English and Mandarin Chinese, 700 each); the language is the prefix of
the clip id (`EN_` or `ZH_`).

How the labels were made (paper, Sec. IV-B): Emilia was labelled automatically,
with Emotion2Vec on the speech and GLM-4-9B-Chat on the text; clips on which the two
agreed were kept, 300 per emotion were drawn at random (4,200 candidates), and four
bilingual experts reviewed every candidate independently; only clips with
unanimous labels from all four were kept. The dataset card says instead that each
clip was checked by "at least two experts".

Labels come from the `emotion` field of `Emo-Emilia-ALL.jsonl`. All seven are
kept; three are written as nouns rather than adjectives:

| `emotion` field | label |
|---|---|
| angry | angry |
| disgusted | disgust |
| fearful | fear |
| happy | happy |
| neutral | neutral |
| sad | sad |
| surprised | surprise |

The `wav` field of the JSONL holds a path from the authors' layout
(`./Emo-Emilia/wav/<id>.wav`); the recipe uses only its file name and reads the
clip from `wav/` of the download.

`create_dataset` converts every clip to 16 kHz mono with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EMO_EMILIA` — the directory holding `Emo-Emilia-ALL.jsonl` and `wav/`, or where to download them to
- `EMO_EMILIA_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `wav/*.wav`, `Emo-Emilia-ALL.jsonl` and `README.md`
from Hugging Face at a pinned commit (`519da7a59f4a5a9b9bba5b5c318f1e2ee0eed97a`)
with `huggingface_hub.snapshot_download`; no login is needed. Files already
present are not fetched again.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-18 against the Hugging Face dataset `ASLP-lab/Emo-Emilia` (repository revision `519da7a59f4a5a9b9bba5b5c318f1e2ee0eed97a`). All counts cover every clip in the release, before any filtering by this recipe.

### Access

Hugging Face, direct download, no sign-in.

### Size of the release

| clips | speakers | total hours |
|---|---|---|
| 1,400 | not identified | 3.29 |

How these were counted: every `.wav` under `wav/` (all 1,400 are listed once in `Emo-Emilia-ALL.jsonl`); hours from the WAV headers (3.00 to 24.55 s). All are 16 kHz mono. By the id prefix, 700 clips are English (1.86 hours) and 700 Mandarin Chinese (1.43 hours).

### Files per label

| `emotion` field | clips | hours |
|---|---|---|
| angry | 200 | 0.529 |
| disgusted | 200 | 0.439 |
| fearful | 200 | 0.408 |
| happy | 200 | 0.503 |
| neutral | 200 | 0.421 |
| sad | 200 | 0.556 |
| surprised | 200 | 0.433 |

How these were counted: the `emotion` field of `Emo-Emilia-ALL.jsonl`. Hours per label are measured in the same way as the total hours above and summed over the clips of that label.

### Issues in the release

- Two pairs of clips have byte-identical content, each with the same label and transcript: `EN_B00017_S08547_W000010` = `EN_B00035_S05775_W000010` (disgusted) and `ZH_B00005_S00050_W000006` = `ZH_B00005_S00084_W000006` (surprised). The second of each pair is listed in `duplicate_ids`.
- The `wav` field of the JSONL gives `./Emo-Emilia/wav/<id>.wav`, a path that does not exist in the release (the clips are under `wav/`).


## Splits

The release is meant as a test set and publishes no partition. To train on it
alone, clips are split 70/10/20 at random within each label (`train_ratio`,
`valid_ratio`, `test_ratio`, `split_seed` in `dataset/config.yaml`). The split is
not stratified by language. Speakers are not identified in the release, so the
splits may share speakers. Because Emo-Emilia is published as an evaluation set,
scores from this recipe are not comparable to results that use the whole set
for testing.

| split | clips | per label |
|---|---|---|
| train | 980 | 140 |
| valid | 140 | 20 |
| test | 280 | 40 |

## License and citation

CC BY-NC 4.0 (non-commercial), by the dataset card. Emo-Emilia is a subset of
Emilia, and the Emo-Emilia card points to Emilia. Emilia itself is distributed on
Hugging Face (`amphion/Emilia-Dataset`) behind Terms of Access, read on 2026-11-25:
Emilia is to be used under CC BY-NC; the researcher accepts full responsibility for
the use and defends and indemnifies the Emilia authors against any claims from it;
colleagues given access must agree to the same terms; access can be terminated at
any time. The Emo-Emilia card does not say whether these terms apply to Emo-Emilia.
Please cite:

```
@misc{zhao2025steeringlanguagemodelstable,
  title         = {Steering Language Model to Stable Speech Emotion Recognition via Contextual Perception and Chain of Thought},
  author        = {Zhixian Zhao and Xinfa Zhu and Xinsheng Wang and Shuiyuan Wang and Xuelong Geng and Wenjie Tian and Lei Xie},
  year          = {2025},
  eprint        = {2502.18186},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  doi           = {10.1109/TASLPRO.2025.3648793},
  url           = {https://arxiv.org/abs/2502.18186},
}
```

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
