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
