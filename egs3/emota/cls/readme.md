# EmoTa speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on EmoTa alone and scores it on a held-out part of EmoTa, classifying each
utterance into one of five emotions.

[EmoTa](https://huggingface.co/datasets/aaivu-labs/EmoTa) is an acted Tamil
emotional speech corpus reflecting Sri Lankan Tamil dialects. According to the
dataset card, 22 native speakers (11 male, 11 female) read 19 semantically
neutral sentences in five emotions (anger, happiness, sadness, fear and
neutral), 936 utterances in all, recorded as 48 kHz mono 16-bit WAV.

The label comes from the file name, `<speaker>_<sentence>_<emotion>.wav` (for
example `01_05_ang.wav`); `data/metadata.csv` is not read. All five codes are
emotions and all are kept:

| code in the file name | label |
|---|---|
| ang | angry |
| hap | happy |
| sad | sad |
| fea | fear |
| neu | neutral |

`create_dataset` converts every recording to 16 kHz mono with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EMOTA` — the directory holding `data/*.wav`, or where to download them to
- `EMOTA_OUTPUT` — where the converted audio and manifests are written

## Download

The dataset is gated. Before the first run, open the
[dataset page](https://huggingface.co/datasets/aaivu-labs/EmoTa) while logged in
to Hugging Face, fill in the form and accept the license terms (access is
granted automatically), then log in on the machine that runs the recipe
(`huggingface-cli login`, or set `HF_TOKEN`). `prepare_source` then downloads
`data/*.wav`, `README.md` and `LICENSE.txt` at a pinned commit
(`58682568acd31be03acf07bb33de3f2528e74563`) with
`huggingface_hub.snapshot_download`. Without access it stops with an error that
explains these steps. Files already present are not fetched again.

## The data

TODO: filled after the first build (counts of the release).

## Splits

The corpus publishes no partition and the dataset card suggests none.
Utterances are split 70/10/20 at random within each label (`train_ratio`,
`valid_ratio`, `test_ratio`, `split_seed` in `dataset/config.yaml`). Speakers and
sentences are not held out, so all three splits share them. Not every speaker
recorded every emotion (the dataset card gives fewer `fear` utterances than the
other emotions), which is why a speaker-held-out split is not used.

## License and citation

EmoTa Academic-Commercial License (EACL) v1.0: academic use under CC BY-NC 4.0,
with additional terms. The data may not be redistributed, sold, sublicensed or
publicly shared, in original or modified form, without written permission of the
licensor; commercial use needs a separate license. See `LICENSE.txt` of the
dataset. Please cite:

```
@inproceedings{thevakumar-etal-2025-emota,
  title     = {{E}mo{T}a: A {T}amil Emotional Speech Dataset},
  author    = {Thevakumar, Jubeerathan and Thavarasa, Luxshan and Sivatheepan, Thanikan and Kugarajah, Sajeev and Thayasivam, Uthayasanker},
  booktitle = {Proceedings of the First Workshop on Challenges in Processing South Asian Languages (CHiPSAL 2025)},
  month     = jan,
  year      = {2025},
  address   = {Abu Dhabi, UAE},
  publisher = {International Committee on Computational Linguistics},
  url       = {https://aclanthology.org/2025.chipsal-1.19/},
  pages     = {193--201},
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
