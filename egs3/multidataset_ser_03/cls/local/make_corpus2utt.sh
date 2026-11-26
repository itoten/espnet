#!/usr/bin/env bash
# Write `corpus2utt` for the up-sampling samplers (`conf/sampler/corpus_pow*.yaml`):
# one line per training corpus, its name and then the keys of its
# utterances in the shape files, which are the line numbers (from 0) of
# `manifest_filtered/train.tsv`. The corpus is the utterance id up to its
# first `-`. Run after `remove_long_short`.
#   local/make_corpus2utt.sh <data_dir>
set -euo pipefail
data=${1:?usage: $0 <data_dir>}
utils=$(cd "$(dirname "$0")/../../../../egs2/TEMPLATE/asr1/utils" && pwd)
awk -F'\t' '{split($1, a, "-"); print NR - 1, a[1]}' "$data/manifest_filtered/train.tsv" \
  > "$data/utt2corpus"
"$utils/utt2spk_to_spk2utt.pl" "$data/utt2corpus" > "$data/corpus2utt"
echo "$(wc -l < "$data/corpus2utt") corpora, $(wc -l < "$data/utt2corpus") utterances"
