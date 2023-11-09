# ExceptionTuning

## Requirements

torch 1.7.0

transformers 4.18.0

## Pretrained Model

Download [codebert-base](https://huggingface.co/microsoft/codebert-base) and [unixcoder-base](https://huggingface.co/microsoft/unixcoder-base), and move the files to ```./microsoft/codebert-base``` and ```./microsoft/unixcoder-base```

## Error Classification (CodeNet)

### Task Definition

The error classification task requires that we assign the same label to a <code, line> pair that have the same run-time error type.
Models are evaluated by accuracy.


### Dataset

We use [CodeNet Python800](https://arxiv.org/abs/2105.12655) dataset on this task.


#### Download and Preprocess

The process for dataset is similar with clone detection unless
```shell
python preprocess_cls.py
```

#### Data Format

After preprocessing dataset, you can obtain the three .jsonl files, 
i.e. `train_exception_cls.jsonl, valid_exception_cls.jsonl, test_exception_cls.jsonl`.

The processed .jsonl files are also at ```./dataset```.

For each file, each line in the uncompressed file represents one function. One row is illustrated below.

   - **code:** the *path* to the source code, for example, `Python800/Project_CodeNet_Python800/p00000/s003971419.py`
   - **label:** the error type of the <code, line> pair
   - **index:** the index of example
   - **line:** the line of code that triggers the error

#### Data Statistics

Data statistics of the dataset are shown in the below table:

ToDo



## Reference

Please cite our work in your publications if it helps your research:
ToDo