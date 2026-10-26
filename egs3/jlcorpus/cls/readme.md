# jlcorpus speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on jlcorpus alone, and scores it on jlcorpus's own test split (10
native classes).

[JL-Corpus](https://github.com/tli725/JL-Corpus) is New Zealand English read
by four actors; five of its ten emotions are the usual primary ones, the
other five secondary (`anxious`/`apologetic`/`assertive`/`concerned`/
`encouraging`). The label comes from the file name; no annotation file is
read. The audio is 44.1 kHz, so `create_dataset` resamples it to 16 kHz.

The corpus is distributed on
[Kaggle](https://www.kaggle.com/datasets/tli725/jl-corpus), which needs an API
token (`pip install kaggle`, then a token from
https://www.kaggle.com/settings/api in `~/.kaggle/access_token` or
`$KAGGLE_API_TOKEN`). `JLCORPUS` points at an existing corpus tree, or where
to download it to.

Seven pairs of clips have the same or nearly the same sound: six are the `a` and `b`
versions of one sentence by `female1` in `excited` (`5a`/`5b`, `6a`/`6b` and `9a`/`9b`,
takes 1 and 2), and one is `male2_neutral_15b_1`/`15b_2`. They are kept as they are.

The corpus publishes no partition, and it has four speakers. One is held out
for test; validation is drawn from the remaining three, because a second
held-out speaker would leave two for training.

## License and citation

The Kaggle listing gives the licence CC0 1.0. The corpus README asks users to cite Jesin
James, Li Tian and Catherine Watson, "An Open Source Emotional Speech Corpus for Human
Robot Interaction Applications", Interspeech 2018. The corpus was evaluated in a
perception test with 120 participants; the survey links are in the README and the
ratings are not bundled.

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

`JLCORPUS_OUTPUT` must already point at the prepared data; `create_dataset`
writes it there if it does not.
