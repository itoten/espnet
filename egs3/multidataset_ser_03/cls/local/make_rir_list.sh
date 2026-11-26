#!/usr/bin/env bash
# Write `<data_dir>/augment/rir.scp` (Kaldi-style `<id> <path>`) from the
# measured room impulse responses of BUT ReverbDB (Szöke et al., 2019;
# CC BY 4.0), unpacked from its RIR-only archive
# (BUT_ReverbDB_rel_19_06_RIR-Only.tgz,
# https://speech.fit.vut.cz/software/but-speech-fit-reverb-database) into
# <reverbdb_dir>. The responses are resampled to the speech rate when they
# are read (CommonPreprocessor's `fs`).
#   local/make_rir_list.sh <reverbdb_dir> <data_dir>
set -euo pipefail
src=${1:?usage: $0 <reverbdb_dir> <data_dir>}
data=${2:?usage: $0 <reverbdb_dir> <data_dir>}
mkdir -p "$data/augment"
src=$(cd "$src" && pwd)
find "$src" -path '*/RIR/*' -name '*.wav' | sort \
  | awk -v root="$src/" '{id = $0; sub(root, "", id); gsub("/", "_", id); sub(".wav$", "", id); print id, $0}' \
  > "$data/augment/rir.scp"
echo "$(wc -l < "$data/augment/rir.scp") RIRs in $data/augment/rir.scp"
