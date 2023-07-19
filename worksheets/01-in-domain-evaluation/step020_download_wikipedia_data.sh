#!/usr/bin/env bash
set -o nounset
set -e

output_dir=output/step020_download_wikipedia_data/

python -m dpr.data.download_data \
  --resource data.wikipedia_split.psgs_w100 \
  --output_dir ${output_dir}