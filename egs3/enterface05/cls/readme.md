# enterface05 speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on enterface05 alone, and scores it on enterface05's own test split
(6 classes: `angry`/`disgust`/`fear`/`happy`/`sad`/`surprise`).

**Read this before using the scores**: subjects did not read a fixed script,
so the words alone give the emotion away; a text-aware frontend can read the
label off the content without modelling prosody. Do not compare this
recipe's numbers with a corpus whose script is fixed across emotions without
saying so.

[eNTERFACE'05](https://enterface.net/enterface05/emotion.html) is an
audio-visual database: the recordings are AVI files carrying 48 kHz stereo
PCM, so `create_dataset` extracts the audio and resamples it to 16 kHz mono.
The site's certificate chain is incomplete, so scripts cannot download it;
fetch `project2_database.zip` by hand and place it under `$ENTERFACE05` (or
let `$ENTERFACE05` hold the already-unpacked tree) for `prepare_source` to
extract. Labels and speakers come from directory names, not file names,
which carry typos.

The release holds 1,293 AVI files in 44 subject directories (the corpus page
reports 42 subjects). 42 subjects have the expected 30 recordings (5 sentences in
each of 6 emotions). Subject 23 lacks 3 (happiness, sentences 1 to 3) and its 27 files
are used. Subject 6 is not segmented: it has one file per emotion, 56 to 107
seconds each (a whole session), against 1.1 to 5.4 seconds for the others, so its 6
files are left out for now. 1,287 recordings from 43 subjects are used.

The corpus publishes no partition. Speaker ids are sorted and dealt out to
train/valid/test (33, 5 and 5 speakers).

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

`ENTERFACE05_OUTPUT` must already point at the prepared data; `create_dataset`
writes it there if it does not.
