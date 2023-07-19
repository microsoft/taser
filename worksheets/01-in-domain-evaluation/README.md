# Tutorial 01: In-Domain Evaluation

This tutorial walks through the steps for evaluating the TASER (initialized by coCondenser-Wiki) on in-domain datasets.
To evaluate the TASER model initialized by BERT-base-uncased, change the `MODEL_NAME` variable in 
[step010_download_model.sh](./step010_download_model.sh),
[step030_generate_dense_embeddings.sh](./step030_generate_dense_embeddings.sh),
and [step050_evaluate_retriever.sh](./step050_evaluate_retriever.sh) 
to `taser-bert-base-uncased`.

| Model                       |  NQ  |  TQ  |  WQ  |  CT  | SQuAD  |
|-----------------------------|:----:|:----:|:----:|:----:|:------:|
| [taser-bert-base-uncased](https://huggingface.co/kelvinih/taser-bert-base-uncased)        | 83.6 | 82.0 | 77.9 | 91.1 |  69.7  |
| [taser-cocondenser-wiki](https://huggingface.co/kelvinih/taser-cocondenser-wiki) | 84.9 | 83.4 | 78.9 | 90.8 |  72.9  |

## Step 1: Download the TASER model

```bash
bash ./step010_download_model.sh
```

## Step 2: Download Wikipedia data 

```bash
bash ./step020_download_wikipedia_data.sh
```

## Step 3: Generate dense embeddings

```bash
bash ./step030_generate_dense_embeddings.sh
```

## Step 4: Download test data 

```bash
bash ./step040_download_test_data.sh
```

## Step 5: Evaluate the dense retriever

```bash
bash ./step050_evaluate_retriever.sh
```

- The top-K accuracy would be printed in the `stdout` file.
