# When Big Models Train Small Ones: Label-Free Model Parity Alignment for Efficient Visual Question Answering using Small VLMs

This repository contains the official code for training, inference, and evaluation of *MPA* from the *EMNLP'25* paper ["When Big Models Train Small Ones: Label-Free Model Parity Alignment for Efficient Visual Question Answering using Small VLMs"]

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

First, for TextVQA you can download the images and respective annotations from their [official website](https://textvqa.org/). Now, on their official website they have provided three splits: train, val, and test, but since answers are not available for questions in the test-set we use the val set(consisting 5,000 questions for 3,166 images) for evaluation purposes. Furthermore, we split the train set(consisting 34,602 questions for 21,953 images) into 85:15 ratio for training and validation purposes. You can access the train, val, and test splits at the following paths:
```
train-split: /data/TextVQA/qwenTrainFormat_train.json
val-split: /data/TextVQA/qwenTrainFormat_eval.json
test-split: /data/TextVQA/TextVQA_0.5.1_val.json
``` 

Second, for STVQA you can download the images and respective annotations from their [official website](https://rrc.cvc.uab.es/?ch=11). Now, on their official website they have provided data(train and test splits) for three(3) tasks. We have provided our results for task 1 only. Now, similarly to TextVQA, STVQA test set also do not have answers so we first split the train set following a 85:15 split to create train+val set and test set(for running evaluation), and then further split the train+val set following a 85:15 split to create train and eval set. You can access the train, val, and test splits at the following paths:
```
train-split: /data/STVQA/QwenTrainFormat_train_task_1_onePerImage_train.json
val-split: /data/STVQA/QwenTrainFormat_train_task_1_onePerImage_eval.json
test-split: /data/STVQA/train_task_1_onePerImage_val.json
```

Third, for ChartQA you can download the images and respective annotations from their [official github repo](https://github.com/vis-nlp/ChartQA). Now, on their github repo they have provided three splits: train, val, and test, and all splits are equipped with labels as well. You can access the train, val, and test splits at the following paths:
```
train-split: /data/ChartVQA/train_onePerImage_QwenFormat_train.json
val-split: /data/ChartVQA/train_onePerImage_QwenFormat_eval.json
test-split: /data/ChartVQA/test_combined.json
```

Fourth, for OK-VQA you can download the images and respective annotations from their [official website](https://okvqa.allenai.org/index.html). Now, on their official website they have provided two splits: train and test, and unlike previous datasets their test-split have the answers as well. So, we simply split the train set following a 85:15 ratio to create train and evaluation sets. You can access the train, val, and test splits at the following paths:
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
Now, in order to identify the parity between the LVLM and SVLM run the following command. Note, you have to pass the path of the PA output json file inside the respective dataloader in PI.py.
```
# run the bash script PI.sh
$ bash PI.sh
```

## Parity Leveler
Now, in this step we will use the Parity samples between LVLM and SVLM generated and identified using PA and PI respectively to train the SVLM to bridge the Parity between the two models. Note, we use the following [github repo](https://github.com/2U1/Qwen2-VL-Finetune?tab=readme-ov-file#full-finetuning) to train the qwen-family models. Also, note you have to pass the train json file generated during the PI step in PL.sh to train on the parity sampled. Run the following command to do the same:
```
# run the bash script PL.sh
$ bash PL/Qwen2-VL-Finetune/scripts/PL.sh
```

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
