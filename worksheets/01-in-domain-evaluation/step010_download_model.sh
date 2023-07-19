#!/usr/bin/env bash

MODEL_NAME="taser-cocondenser-wiki"

output_dir=output/step010_download_model/${MODEL_NAME}
mkdir -p ${output_dir}

wget -O ${output_dir}/pytorch_model.bin https://huggingface.co/kelvinih/${MODEL_NAME}/resolve/main/pytorch_model.bin 