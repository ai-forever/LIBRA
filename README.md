# LIBRA: Long Input Benchmark for Russian Analysis

![logo](https://i.imgur.com/uZYPoc9.jpeg)
 
## Introduction

Welcome to the official GitHub repository for **LIBRA (Long Input Benchmark for Russian Analysis)**. This repository contains the codebase and documentation for evaluating the capabilities of large language models (LLMs) in understanding and processing long texts in Russian.

## Datasets

LIBRA includes 21 datasets adapted for different tasks and complexities. The datasets are divided into four complexity groups and allow evaluation across various context lengths ranging from 4,000 to 128,000 tokens.

## Usage

### Adding Your Own Dataset

In order to add your own dataset, follow these steps:

  1. Add your dataset to the configs/datasets_config.json file.
  2. Create a config file for it (e.g., longchat32k.ini) and specify the necessary parameters in it.

### Generating Answers


To run the script to generate answers to the tasks, use the following command:

```bash
python main.py -c path_to_config
```

### Metric Evaluation

For metric evaluation, use the following command:

```bash
python eval.py -p path_to_predictions
```

## License

This project is licensed under the MIT License.
