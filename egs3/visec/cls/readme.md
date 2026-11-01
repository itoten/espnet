# ViSEC speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on ViSEC alone and scores it on ViSEC's own test split, classifying each
utterance into one of four emotions.

ViSEC is a Vietnamese speech emotion corpus released with Pham Viet Thanh et al.,
"A Robust Pitch-Fusion Model for Speech Emotion Recognition in Tonal Languages"
(ICASSP 2024, pp. 12386-12390). The copy used here has 5,280 rows (3.18 hours) with the
emotions `angry`, `happy`, `neutral` and `sad`, a gender (`female`/`male`), an
accent (`south`/`north`/`mid`) and a `speaker_id` per row. The clips are 16 kHz mono
(5,178 are 16-bit and 102 are 24-bit PCM) and 1.0 to 24.7 seconds long.

The label is the `emotion` column. The materials say nothing about how the
recordings were made or labelled.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Hugging Face dataset `Dennis290/ViSEC` (repository revision `70f4b62f432d85c0b52dee3855a5f8dd244f093a`, last modified 2026-09-08; `data/train-00000-of-00001.parquet`, downloaded around 2026-10-15). This is a re-publication by another account; the original `hustep-lab/ViSEC` answered HTTP 401. All counts cover every sample in the release, before any filtering by this recipe.

### Access

Hugging Face, direct download, no sign-in. The original repository `hustep-lab/ViSEC` could not be accessed (HTTP 401).

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 5,280 | 147 | 3.18 |

How these were counted: every row of the parquet file (16 kHz mono; 5,178 clips 16-bit and 102 clips 24-bit PCM; 1.0 to 24.7 s); hours from the clip lengths. Speakers are the 147 distinct `speaker_id` values (0 to 146; ID 0 alone has 2,217 rows, 42%). The release also has gender (`female` / `male`) and accent (`south` / `north` / `mid`) columns.

### Files per label

| label (the `emotion` column) | samples |
|---|---|
| neutral | 1,507 |
| angry | 1,466 |
| happy | 1,228 |
| sad | 1,079 |

How these were counted: the `emotion` column.

### Issues in the release

- 277 groups of byte-identical clips; in 3 groups the emotion differs between the copies, and in 8 groups the speaker ID differs.
- Within one speaker ID the metadata is inconsistent: 2 IDs mix genders and 6 IDs mix accents (1 to 3 rows with another value).
- The materials say nothing about how the recordings were made or labelled; Whisper identified Vietnamese in 189 of 200 transcribed clips.
- The paper's abstract (read; the full text was not) says the data come from public media.


## Provenance

The original repository (`hustep-lab/ViSEC` on Hugging Face) answers 401. The recipe
uses a **re-upload under a different account**, `Dennis290/ViSEC` (pinned to commit
`70f4b62f432d85c0b52dee3855a5f8dd244f093a`, not gated, one 367 MB parquet file,
SHA-256 checked, created 2026-09-08). Whether this account belongs to the authors, and
whether the copy equals the original, is unknown, so it may be an unofficial mirror.
Its card says `cc-by-4.0` and asks to cite the paper above; the terms of the original
release could not be checked. The paper's abstract says the data were collected from
public media, so the rights in the audio are also unclear.

**The corpus is on hold**: use it only for local experiments until the official source
is confirmed and the licence is clear.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `VISEC` — the directory holding `train-00000-of-00001.parquet`, or where to download it to
- `VISEC_OUTPUT` — where the converted audio and manifests are written

## Data cleaning

The parquet file has 277 groups of byte-identical clips (276 pairs and one triple,
278 surplus rows), which would leak between splits, so each is kept once. In three
of these groups the copies carry different emotions (for example `sad` and
`angry`); those six rows are dropped. That leaves 4,999 distinct clips (2.96 hours).

## Download

`prepare_source` downloads the parquet file and checks its SHA-256 (31 seconds
on one run). `create_dataset`, `remove_long_short` and `prepare_labels`
together took about 4 minutes (create_dataset alone 222 s, almost all of it
converting the 4,999 clips with ffmpeg).

## Splits

No partition is published. Whole `speaker_id` groups are kept in one split, so no
`speaker_id` appears in two splits. The id `0` alone covers 2,217 of the 5,280 rows (42%); all of them are female and
south accent, with all four emotions. Whether any `speaker_id` is a real speaker
is not documented. A group larger than the test quota (20% of the utterances) is
therefore always put in train, and the other groups are shuffled (`split_seed`
in `dataset/config.yaml`) and fill test, then valid, then train. Labels are not
balanced across the splits.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 3,422 | 3,422 |
| valid | 514 | 514 |
| test | 1,063 | 1,063 (not filtered) |

Train holds 1,160 angry, 938 neutral, 689 sad and 635 happy utterances; valid
138 angry, 160 neutral, 155 sad and 61 happy; test 167 angry, 407 neutral, 212
sad and 277 happy. The 4,999 distinct clips are 147 `speaker_id` groups.

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
