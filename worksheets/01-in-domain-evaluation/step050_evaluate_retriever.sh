#!/usr/bin/env bash
set -o nounset
set -e

MODEL_NAME="taser-cocondenser-wiki"

basedir=$(pwd -P)

path_to_model="${basedir}/output/step010_download_model/${MODEL_NAME}/pytorch_model.bin"
path_to_wikipedia_tsv="${basedir}/output/step020_download_wikipedia_data/downloads/data/wikipedia_split/psgs_w100.tsv"
ctx_dir="${basedir}/output/step030_generate_dense_embeddings"
data_dir="${basedir}/output/step040_download_test_data"

output_dir="${basedir}/output/step050_evaluate_retriever"
mkdir -p ${output_dir}

for dataset in nq-test trivia-test webq-test curatedtrec-test squad1-test; do
  echo "evaluating ${dataset}..."

  dataset_underscore=$(echo ${dataset} | tr '-' '_')

  export MKL_THREADING_LAYER=GNU
  python -m taser.evaluate_dense_retriever \
    model_file=${path_to_model} \
    qa_dataset=[${dataset_underscore}] \
    ctx_datatsets=[dpr_wiki] \
    ctx_sources.dpr_wiki.file=${path_to_wikipedia_tsv} \
    datasets.${dataset_underscore}.file=${data_dir}/${dataset}/downloads/data/retriever/qas/${dataset}.csv \
    encoder.shared_encoder=True \
    encoder.use_moe=True \
    encoder.moe_type="mod3:fwd" \
    encoder.num_expert=2 \
    encoder.use_infer_expert=False \
    encoder.per_layer_gating=False \
    encoder.factor_rep=False \
    encoder.mean_pool=False \
    norm_vector=False \
    encoded_ctx_files=["${ctx_dir}/wiki_passages_*"] \
    out_file=["${output_dir}/${dataset}.results.json"] \
    1> ${output_dir}/${dataset}.stdout \
    2> ${output_dir}/${dataset}.stderr
done
