#!/usr/bin/env bash
set -o nounset
set -e

basedir=$(pwd -P)

output_dir="${basedir}/output/step040_download_test_data"

for dataset in nq-test trivia-test webq-test curatedtrec-test squad1-test; do
  python -m dpr.data.download_data \
    --resource data.retriever.qas.${dataset} \
    --output_dir ${output_dir}/${dataset}
done