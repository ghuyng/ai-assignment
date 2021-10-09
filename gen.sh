#!/usr/bin/env bash
set -euo pipefail

mkdir -p data

for i in $(seq $1 $2); do
	out=data/round${i}.txt
	# Generate the round when it doesn't exists
	[[ -f out ]] || python -m main >>$out
done
