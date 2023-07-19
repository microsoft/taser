#!/usr/bin/env bash
set -o nounset
set -e


MODEL_NAME="taser-cocondenser-wiki"

basedir=$(pwd -P)

path_to_model="${basedir}/output/step010_download_model/${MODEL_NAME}/pytorch_model.bin"
path_to_wikipedia_tsv="${basedir}/output/step020_download_wikipedia_data/downloads/data/wikipedia_split/psgs_w100.tsv"

output_dir="${basedir}/output/step030_generate_dense_embeddings"
mkdir -p ${output_dir}

log_dir="${output_dir}/logs"
mkdir -p ${log_dir}

num_shards=16

for shard_id in $(seq 0 ${num_shards}); do
  echo "generating dense embeddings for shard ${shard_id}..."

  MKL_THREADING_LAYER=GNU
  python -m taser.generate_dense_embeddings_v2 \
    model_file=${path_to_model} \
    ctx_sources.dpr_wiki.file=${path_to_wikipedia_tsv} \
    ctx_src=dpr_wiki \
    norm_vector=False \
    encoder.mean_pool=False \
    encoder.shared_encoder=True \
    encoder.use_moe=True \
    encoder.moe_type="mod3:fwd" \
    encoder.num_expert=2 \
    encoder.use_infer_expert=False \
    encoder.per_layer_gating=False \
    encoder.factor_rep=False \
    batch_size=1096 \
    fp16=True \
    shard_id=${shard_id} \
    num_shards=${num_shards} \
    gpu_id=0 \
    num_gpus=1 \
    out_file=${output_dir}/wiki_passages \
    1> ${log_dir}/shard${shard_id}.stdout \
    2> ${log_dir}/shard${shard_id}.stderr
  done
