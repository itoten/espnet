# Multi-corpus speech emotion recognition, out-of-domain evaluation (ood20)

Trains the `cls` system (a pre-trained speech encoder + attentive
statistics pooling + a fully connected head) on ten corpora used in full, and
scores it on corpora it never sees. The encoder can be swapped; everything
else stays the same.

| Role | Corpora | Utterances |
|---|---|---:|
| Training (every split of each corpus) | CREMA-D, EMNS, EmoDB, Emozionalmente, eNTERFACE'05, JL-Corpus, MELD, Quechua Collao, SUBESCO, Thorsten | 55,524 (55,086 after the 0.5–30 s filter) |
| Evaluation (every split of each corpus) | EmoV-DB, IEMOCAP, TESS, SAVEE, RAVDESS | 17,184 |
| Evaluation | MSP-Podcast Test1 / Test2 | 35,728 / 11,363 |

Each corpus is read through its own `egs3/<corpus>/emotion` recipe, from the
unfiltered `manifest/` that recipe writes. Utterance counts are those
manifests' row counts. The MSP-Podcast recipe's manifests carry seven labels:
angry, disgust, fear, happy, neutral, sad and surprise.

All 20 labels found in the training corpora are kept as classes
(`dataset/config.yaml`'s `label_map`). Every evaluation label is one of them.

## Encoders

Each encoder has a frontend config in `conf/frontend/`, and a training and an
inference config that combine it with the shared `conf/training.yaml` and
`conf/inference.yaml`.

| `<encoder>` | Model | Frontend | Input |
|---|---|---|---|
| `wavlm_large` | `microsoft/wavlm-large` | `Wav2Vec2Frontend` | waveform |
| `hubert_large` | `facebook/hubert-large-ll60k` | `Wav2Vec2Frontend` | waveform |
| `wav2vec2_large` | `facebook/wav2vec2-large-lv60` | `Wav2Vec2Frontend` | waveform |
| `xls_r_300m` | `facebook/wav2vec2-xls-r-300m` | `Wav2Vec2Frontend` | waveform |
| `w2v_bert_2` | `facebook/w2v-bert-2.0` | `W2vBertFrontend` | 80-dim filterbanks, two frames stacked |
| `whisper_medium` | `openai/whisper-medium` (encoder) | `WhisperFrontend` | 80-dim log-mel, padded to 30 s |
| `owsm_v4_medium` | `espnet/owsm_v4_medium_1B` (encoder) | `OwsmFrontend` | 128-dim log-mel, padded to 30 s |

Shared by every encoder: the data, the pooling and head, the loss, AdamW at
lr 1e-5 with an effective batch of 24, 30 epochs, seed 0, a layer drop of 0.1,
and frozen convolutions in front of the encoder layers. Time masking at
`mask_time_prob: 0.075` applies to the encoders that have it (the
wav2vec 2.0 family and w2v-BERT 2.0); Whisper and OWSM have none.

What differs is the input each encoder was pre-trained on. The wav2vec 2.0
family gets the waveform standardised with the training set's statistics.
w2v-BERT 2.0, Whisper and OWSM compute their own features with their own
extractors. Whisper's encoder only accepts 30 s of input, and OWSM is trained
and decoded on 30 s windows, so both are fed waveforms padded to 30 s; only the
frames of the real waveform are pooled.

## How this recipe differs from the other multi-corpus recipes

- **No validation during training.** `trainer.limit_val_batches: 0`. The
  `valid` split is a 64-row placeholder taken from the training manifest,
  present only because `DataOrganizer` requires one.
- **Fixed length.** 30 epochs without early stopping; the model after the last
  epoch (`exp/<tag>/last.ckpt`) is the one scored.

## Running

Run each line as its own command. Under multi-GPU DDP, Lightning re-runs the
whole command once per GPU, so combining data preparation or inference with
`train` repeats them. Replace `wavlm_large` with any `<encoder>` above; data
preparation is shared and only needs to run once.

```bash
python run.py --stages create_dataset remove_long_short prepare_labels collect_stats \
    --training_config conf/training_wavlm_large.yaml
python run.py --stages train --training_config conf/training_wavlm_large.yaml
python run.py --stages infer measure \
    --training_config conf/training_wavlm_large.yaml \
    --inference_config conf/inference_wavlm_large.yaml \
    --metrics_config conf/metrics.yaml
```

Results go to `exp/train_<encoder>_ood20/inference/`.

To resume an interrupted training run:

```bash
RESUME_CKPT=exp/train_wavlm_large_ood20/last.ckpt \
    python run.py --stages train --training_config conf/training_wavlm_large.yaml
```

## Scoring

`conf/metrics.yaml` reuses the five `cls` metrics. The model predicts
over all 20 classes; UA and MacroF1 average over the classes that appear in
each evaluation set's references.
