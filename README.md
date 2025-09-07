# Official Implementation of our EMLNP 2025 Paper "When Big Models Train Small Ones: Label-Free Model Parity Alignment for Efficient Visual Question Answering using Small VLMs"

[![GitHub stars](https://img.shields.io/github/stars/vl2g/MPA.svg?style=social&label=Star&maxAge=2592000)](https://github.com/vl2g/MPA/) ![Visitors](https://visitor-badge.laobi.icu/badge?page_id=vl2g.MPA)


This repository contains the official code for training, inference, and evaluation of **Model Parity Aligner (MPA)**.

## To setup environment
```
# create new docker container(using the mentioned docker image)
$ docker run -it -d --name MPA --gpus=all -v <path-to-your-directory>:/workspace pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# create new env MPA
$ conda create -n MPA python=3.13.5

# activate MPA
$ conda activate MPA

# install dependencies
$ pip install -r requirements.txt
```

## Dataset
Now, we show the results of MPA on four datasets namely, TextVQA, STVQA, ChartQA, and OKVQA. Please follow the following instructions to successfully create the splits used for each dataset.

First, for TextVQA you can download the images and respective annotations from their [official website](https://textvqa.org/). You can access the train, val, and test splits at the following paths:
```
train-split: /data/TextVQA/qwenTrainFormat_train.json
val-split: /data/TextVQA/qwenTrainFormat_eval.json
test-split: /data/TextVQA/TextVQA_0.5.1_val.json
``` 

Second, for STVQA you can download the images and respective annotations from their [official website](https://rrc.cvc.uab.es/?ch=11). You can access the train, val, and test splits at the following paths:
```
train-split: /data/STVQA/QwenTrainFormat_train_task_1_onePerImage_train.json
val-split: /data/STVQA/QwenTrainFormat_train_task_1_onePerImage_eval.json
test-split: /data/STVQA/train_task_1_onePerImage_val.json
```

Third, for ChartQA you can download the images and respective annotations from their [official github repo](https://github.com/vis-nlp/ChartQA). You can access the train, val, and test splits at the following paths:
```
train-split: /data/ChartVQA/train_onePerImage_QwenFormat_train.json
val-split: /data/ChartVQA/train_onePerImage_QwenFormat_eval.json
test-split: /data/ChartVQA/test_combined.json
```

Fourth, for OK-VQA you can download the images and respective annotations from their [official website](https://okvqa.allenai.org/index.html). You can access the train, val, and test splits at the following paths:
```
train-split: /data/OKVQA/okvqa_QwenFormat_train.json
val-split: /data/OKVQA/okvqa_QwenFormat_eval.json
test-split: /data/OKVQA/okvqa_val_combine.json
```

## Pseudo Annotator (PA)
Now, in order to generate Pseudo Annotation of unlabeled images for task 'T', run the following command. This will create a new directory(if one does not already exists) inside the scripts directory and dump the PA json files further inside a directory following the date on which the experiment is being run. Note, demo files for the sake of demonstration are already present in the results directory.
```
# change to scripts dir
$ cd scripts/

# run the bash script PA.sh
$ bash PA.sh
```

## Parity Identifier (PI)
This is the module that is responsible to identify samples that represent the knowledge gaps between S-VLM and L-VLM. Note, you have to pass the path of the PA output json file inside the respective dataloader in PI.py.
```
# run the bash script PI.sh
$ bash PI.sh
```

## Parity Leveler
Now, Parity samples obtained by PI module are used to train the SVLM to enhance it. Also, note you have to pass the train json file generated during the PI step in PL.sh to train on the parity samples. Run the following command to do the same:
```
# run the bash script PL.sh
$ bash PL/Qwen2-VL-Finetune/scripts/PL.sh
```
Note, we use the following [github repo](https://github.com/2U1/Qwen2-VL-Finetune?tab=readme-ov-file#full-finetuning) to train the qwen-family models.

## Evaluate
Now, to evaluate pre-trained and MPA trained models you can run the following command:
```
# run the bash script evaluate.sh
$ bash evaluate.sh
```

## License
This code and data are released under the [MIT license](LICENSE.txt).

## Acknowledgements
1. We used code-base and pre-trained models of [Qwen2vl](https://github.com/QwenLM/Qwen2.5-VL).
