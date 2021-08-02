#!/usr/bin/env bash
set -euo pipefail

mkdir data

for i in {$1..$2}; do
	out=data/round${i}.txt
	# Generate the round if not already exists
	[[ -f out ]] || python -m main >>data/round${i}.txt
done
