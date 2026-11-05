# thorsten speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on thorsten alone, and scores it on thorsten's own test split (8
classes: `amused`/`angry`/`disgust`/`drunk`/`neutral`/`sleepy`/`surprise`/
`whisper`, spelled to match the other corpora here; the corpus's own spelling
is `disgusted`/`surprised`).

[Thorsten-Emotional](https://www.openslr.org/110/) is one German speaker
reading the same 300 sentences eight times over, once per label; three of the
eight (`drunk`, `sleepy`, `whisper`) are speaking styles rather than emotions.
The label comes from the directory name. The audio is 22.05 kHz, so
`create_dataset` resamples it to 16 kHz. `THORSTEN` points at an existing
corpus tree, or where to download it to.

One speaker, so a speaker-independent split cannot be built and scores from
this recipe are speaker-dependent. The split works on sentences instead: all
eight recordings of a sentence go to the same split, so the same German text
never appears in both train and test.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the OpenSLR resource 110 (Thorsten-Emotional v02; Zenodo record 5525023; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from OpenSLR, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 2,399 | 1 | 2.93 |

How these were counted: every `.wav` in the archive (22.05 kHz mono 16-bit, 1.15 to 12.38 s); hours from the file lengths. One German speaker reads the same 300 sentences in eight ways.

### Files per label

| label (directory name in the release) | samples | hours |
|---|---|---|
| amused | 300 | 0.306 |
| angry | 300 | 0.339 |
| disgusted | 300 | 0.384 |
| drunk | 300 | 0.422 |
| neutral | 300 | 0.309 |
| sleepy | 300 | 0.507 |
| surprised | 300 | 0.301 |
| whisper | 299 | 0.365 |

How these were counted: the directory each file is in. `drunk`, `sleepy` and `whisper` are speaking styles, not emotions. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- The description lists 300 recordings per style (2,400 in all); the `whisper` directory holds 299.
- The release has no listening test.


## Data and licence

The corpus is one German speaker reading the same 300 sentences in 8 ways. The release
used here has 2,399 recordings (the `whisper` directory has 299, the description says
300 per style; 2,400 in all), 22.05 kHz mono 16-bit, about 2.9 hours. `drunk`, `sleepy`
and `whisper` are used as classes like the others. The licence is CC0 1.0 (Zenodo
[5525023](https://zenodo.org/records/5525023), also OpenSLR 110). There is no listening
test.

## Quick start

```bash
# 1) Write the manifests
python run.py --stages create_dataset remove_long_short prepare_labels \
    --training_config conf/training.yaml

# 2) Collect feature statistics
python run.py --stages collect_stats \
    --training_config conf/training.yaml

# 3) Train
python run.py --stages train \
    --training_config conf/training.yaml

# 4) Infer
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 5) Score
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`THORSTEN_OUTPUT` must already point at the prepared data; `create_dataset`
writes it there if it does not.
