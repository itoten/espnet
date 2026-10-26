# emozionalmente speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on emozionalmente alone, and scores it on emozionalmente's own test
split (7 classes: `angry`/`disgust`/`fear`/`happy`/`neutral`/`sad`/
`surprise`, spelled to match the other corpora here; the corpus's own
spelling is `anger`/`joy`/`neutrality`/`sadness`).

[Emozionalmente](https://doi.org/10.5281/zenodo.12616095) is 6,902 Italian
utterances crowdsourced from 431 non-professional actors, already 16 kHz
mono WAV. `EMOZIONALMENTE` points at an existing corpus tree, or where to
download it to; the Zenodo archive is checked against its published MD5.

The corpus publishes a speaker-independent train/dev/test split, stratified
by emotion, gender and age, and this recipe reads it from
`metadata/split/*.csv` rather than rebuilding it.

## Data and ratings

The 6,902 clips (0.76 to 13.9 seconds, 3.81 seconds on average, 16 kHz mono) were acted
by 431 amateur actors, each saying 18 sentences with the seven emotional intentions,
and were mostly recorded with non-professional equipment. The label is the intention
of the actor, not a vote. Besides the audio, the corpus holds the demographics of the
actors and of the 829 evaluators, and the 34,510 evaluations (five per clip, with an
audio-quality flag). A listener's single answer equals the intention for 66.4% of the
judgements and the strict majority of the five equals it for 5,025 of 6,902 clips; the
ratings are not used here.

## License and citation

The Zenodo record
[10.5281/zenodo.12616095](https://doi.org/10.5281/zenodo.12616095) is licensed CC BY 4.0.
Cite F. Catania, J. W. Wilke and F. Garzotto, "Emozionalmente: A Crowdsourced Corpus of
Simulated Emotional Speech in Italian", IEEE Transactions on Audio, Speech and Language
Processing 33, pp. 1142-1155, 2025.

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

`EMOZIONALMENTE_OUTPUT` must already point at the prepared data;
`create_dataset` writes it there if it does not.
