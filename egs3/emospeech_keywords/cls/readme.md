# EmoSpeech (keyword) speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on EmoSpeech (keyword) alone and scores it on its own test split, classifying
each utterance into one of four emotions.

The Indian EmoSpeech Command dataset ([Banga et al., 2019](https://arxiv.org/abs/1910.13801))
holds short spoken commands (`help`, `bacho`, `stop`, `no`, `go`, `yes`, plus
`unknown` and `background` clips) recorded with an emotion and an environment tag.
The Kaggle archive has 7,735 wav files in folders named after the recording place.
The label is the emotion in the file name
`<keyword>-<environment>-<emotion>-<stamp>-<user hash>.wav`.

After removing 90 duplicate recordings (identical md5) and 164 clips tagged `none`,
7,481 utterances remain, all 16 kHz mono and 1.96 to 3.75 seconds long, 6.34 hours
in total. The four labels are `angry`, `calm`, `fear` (`Fearful` in the file names)
and `happy`.

Environment variables (both optional; defaults are `download/` and `data/` under the
recipe):

- `EMOSPEECH_KEYWORDS` — the directory holding the extracted `emospeech-dataset-v1/`
- `EMOSPEECH_KEYWORDS_OUTPUT` — where the copied audio and manifests are written

## Download

The Kaggle dataset ([ujjwal9/emospeech](https://www.kaggle.com/datasets/ujjwal9/emospeech))
needs a Kaggle login, so the recipe does not download it. Download the zip by hand,
extract it and point `$EMOSPEECH_KEYWORDS` at the folder that holds
`emospeech-dataset-v1/`; otherwise `create_dataset` stops with the URL. On one run
`create_dataset`, which copies the audio, took 342 seconds. The download time was
not measured.

## Splits

The archive publishes no partition. 5,709 of the 7,735 recordings carry the single
"unknown user" hash, so speakers cannot be held out. Utterances are split 70/10/20 at
random within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed`
in `dataset/config.yaml`), so the splits share speakers and scores are optimistic.

| split | utterances | hours | after `remove_long_short` | angry | calm | fear | happy |
|---|---|---|---|---|---|---|---|
| train | 5,238 | 4.44 | 5,238 | 778 | 2,654 | 1,122 | 684 |
| valid | 748 | 0.63 | 748 | 111 | 379 | 160 | 98 |
| test | 1,495 | 1.27 | 1,495 (not filtered) | 222 | 758 | 320 | 195 |

## License and citation

The archive is distributed under CC BY-NC-SA 4.0 (`License.txt` in the archive), so
use is non-commercial and derivatives must be shared alike. Please cite

```
@misc{1910.13801,
  Author = {Subham Banga and Ujjwal Upadhyay and Piyush Agarwal and Aniket Sharma and Prerana Mukherjee},
  Title = {Indian EmoSpeech Command Dataset: A dataset for emotion based speech recognition in the wild},
  Year = {2019},
  Eprint = {arXiv:1910.13801},
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
