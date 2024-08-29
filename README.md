# LIBRA: Long Input Benchmark for Russian Analysis

<p align="center">
  <picture>
    <img alt="LIBRA" src="docs/LIBRA_logo.png" style="max-width: 100%;">
  </picture>
</p>
 
## Introduction

Welcome to the official GitHub repository for **LIBRA (Long Input Benchmark for Russian Analysis)**. This repository contains the codebase and documentation for evaluating the capabilities of large language models (LLMs) in understanding and processing long texts in Russian.

## Usage

### Adding Your Own Model

In order to add your own model, create a config file using configs/template.ini for it (e.g., longchat32k.ini) and specify the necessary parameters in it.

### Generating Answers

First, you need to generate answers for each task, to do this, use the following command:

```bash
python predict.py -c path_to_config
```

The predictions will be saved in "predictions/" or wherever you choose in your config.

### Metric Evaluation

After the generated predictions are saved, you need to run the command to evaluate:

```bash
python eval.py -p path_to_predictions
```

The results will be saved in "results/".

## Datasets

LIBRA includes 21 datasets adapted for different tasks and complexities. The datasets are divided into four complexity groups and allow evaluation across various context lengths ranging from 4k to 128k tokens.

<p align="center">
  <picture>
    <img alt="LIBRA" src="docs/LIBRA_table.png" style="max-width: 100%;">
  </picture>
</p>

### Tasks and Complexity Groups

#### Group I: Simple Information Retrieval

- **Passkey**: Extract a relevant piece of code number from a long text fragment. Based on the original [PassKey test](https://github.com/CStanKonrad/long_llama/blob/main/examples/passkey.py) from the m LongLLaMA’s GitHub repo.
- **PasskeyWithLibrusec**: Similar to Passkey but with added noise from Librusec texts.

#### Group II: Question Answering and Multiple Choice

- **MatreshkaNames**: Identify the person in dialogues based on the discussed topic. We used [Matreshka](https://huggingface.co/datasets/zjkarina/matreshka) dataset and [Russian Names](https://www.kaggle.com/datasets/rai220/russian-cyrillic-names-and-sex/data) dataset to create this and the next task.
- **MatreshkaYesNo**: Indicate whether a specific topic was mentioned in the dialog.
- **LibrusecHistory**: Answer questions based on historical texts. Ideologically similiar to the [PassageRetrieval dataset](https://huggingface.co/datasets/THUDM/LongBench/viewer/passage_retrieval_en) from LongBench.
- **ruTREC**: Few-shot in-context learning for topic classification. Created by translating the [TREC dataset](https://huggingface.co/datasets/THUDM/LongBench/viewer/trec_e) from LongBench.
- **ruSciFi**: Answer true/false based on context and general world knowledge. Translation of [SciFi dataset](https://huggingface.co/datasets/L4NLP/LEval/viewer/sci_f) from L-Eval which originally was based on [SF-Gram](https://github.com/nschaetti/SFGram-dataset).
- **ruSciAbstractRetrieval**: Retrieve relevant paragraphs from scientific abstracts.
- **ruTPO**: Multiple-choice questions similar to TOEFL exams. Translation of the [TPO dataset](https://huggingface.co/datasets/L4NLP/LEval/viewer/tpo) from L-Eval.
- **ruQuALITY**: Multiple-choice QA tasks based on detailed texts. Created by translating the [QuALITY dataset](https://huggingface.co/datasets/L4NLP/LEval/viewer/quality) from L-Eval.

#### Group III: Multi-hop Question Answering

- **ruBABILongQA**: 5 long-context reasoning tasks for QA using facts hidden among irrelevant information.
- **LongContextMultiQ**: Multi-hop QA based on Wikidata and Wikipedia.
- **LibrusecMHQA**: Multi-hop QA requiring information distributed across several text parts.
- **ru2WikiMultihopQA**: Translation of the [2WikiMultihopQA dataset](https://huggingface.co/datasets/THUDM/LongBench/viewer/2wikimqa_e) from LongBench.

#### Group IV: Complex Reasoning and Mathematical Problems

- **ruSciPassageCount**: Count unique paragraphs in a long text. Uses the basic idea of the original [PassageCount dataset](https://huggingface.co/datasets/THUDM/LongBench/viewer/passage_count) from LongBench.
- **ruQasper**: Question Answering over academic research papers. Created by translating the [Qasper dataset](https://huggingface.co/datasets/THUDM/LongBench/viewer/qasper_e) from LongBench.
- **ruGSM100**: Solve math problems using Chain-of-Thought reasoning. Created by translating the [GSM100](https://huggingface.co/datasets/L4NLP/LEval/viewer/gsm100) dataset from L-Eval.

## Citation

```
@misc{churin2024longinputbenchmarkrussian,
      title={Long Input Benchmark for Russian Analysis},
      author={Igor Churin and Murat Apishev and Maria Tikhonova and Denis Shevelev and Aydar Bulatov and Yuri Kuratov and Sergei Averkiev and Alena Fenogenova},
      year={2024},
      eprint={2408.02439},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.02439},
}
```
