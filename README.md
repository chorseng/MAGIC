# MAGIC

Official Implementation of "Multimodal Dialog System: Generating Responses via Adaptive Decoders".

## Data Preparation

1. [Download](https://drive.google.com/drive/folders/1s0n3thHE2UYCjF9-tDDmBqdxA1XjWFyV) the dialog dataset (preprocessed by us) and the knowledge base.
2. [Download](https://github.com/ChenTsuei/UMD) the crawled images.

## Platform

MAGIC was developed and tested on Linux (Ubuntu 18.04 x64). Theoretically, Unix-like OS should work just fine. It's known that Windows is a bit more involved to get running.

## Prerequisite

- Python 3.7+
- CUDA 8.0+

Python 3.7+ is required because of the widely used type annotation, as specified by PEP 484 and PEP 526.

### Python Package

- Pytorch 1.0
- NLTK 3.4
- PIL 5.3.0

The above packages can be installed using the Python package managers like `pip` or `conda`.

## Configuration

Before start, the configuration file `config/dataset_config.py` is supposed to be modified first.

`data_directory` is the root of the data directory, and the directory is supposed to look like the following structure. You should download the data and organize it according to this structure.

```
<data_directory>
├── dialogs
│   ├── train
│   │   ├── <train_dialog_file_name>.json
│   │   └── ...
│   ├── valid
│   │   ├── <valid_dialog_file_name>.json
│   │   └── ...
│   └── test
│   │   ├── <test_dialog_file_name>.json
│   │   └── ...
├── images
│   ├── <image_file_name>.jpg
│   └── ...
├── knowledge
│   ├── styletip
│   │   └── styletips_synset.txt
│   ├── celebrity
│   │   └── celebrity_distribution.json
│   └── products
│       ├── <product_file_name>.json
│       └── ...
├── glove.txt
└── url2img.txt
```

> Of course, the data directory can be organized in the way you like, as long as the configuration variables are changed accordingly.

`dump_dir` is the directory of the extracted data file and the saved model of each task. `dump_dir` should be created manually. It will look like this.

```
<dump_dir>
├── common_raw_data.pklknowledge_data
├── knowledge_data.pkl
├── train_raw_data.pkl
├── valid_raw_data.pkl
├── test_raw_data.pkl
├── intention_train_dialog.pkl
├── intention_valid_dialog.pkl
├── intention_test_dialog.pkl
└── ...
```

## Data Processing

If you want to extract the dialog data for all tasks (`intention`, `text`, `recommend`, `knowledge_styletip`, `knowledge_attribute`, `knowledge_celebrity`) and all modes (`train`, `valid`, `test`), simply run `preprocess_data.py`. Or you can specify tasks and modes manually as follows.

```
./preprocess_data.py --<task> [modes]
```

For example, if you want to extract the data for `text` and `recommend` task, you can use the following command.

```
./preprocess_data.py --text --recommend
```

which is a shortcut of

```
./preprocess_data.py --text train valid test --recommend train valid test
```

If you want to extract the train and valid data for text task only, simply run:

```
./preprocess_data.py --text train valid
```

If you want to extract raw data only, you can use the following command.

```
./preprocess_data.py --raw_data
```

## Train

If you are using NVIDIA CUDA, you can run `train.sh <gpu_id> <task_name> <model_file> <log_file>` to train the model of the specified task (`intention`, `text`, `recommend`, `knowledge_styletip`, `knowledge_attribute`, `knowledge_celebrity`) as a daemon, the output will be stored into `<log_file>`.

If you run knowledge-based task (`knowledge_*`) for the first time, the knowledge data will be processed immediately. After that, the processed knowledge data will be read directly if you run knowledge-based tasks again. So the recommended way is to run other knowledge-based tasks after finishing the knowledge data processing procedure.

The evaluation procedure will run automatically when out of patience (valid loss didn't decrease for several times). The patience can be set in the train configuration file.

The evaluation results for text generation task (`text` / `knowledge_*`) will be stored into `<task_name>.out`. 

## Evaluation

Perl script [mteval-v14.pl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl) is used to evaluate the result of the text generation task (`text` / `knowledge_*`).
Or you can just run `eval.sh` after the evaluation of `text` and `knowledge_*` tasks (There is supposed to be `text.out` and `knowledge_*.out` in the root directory of the code).

> The result is not reliable if there's any exception running `eval.sh`.

## Known issues

There're some known issues in the code:
- There might be some bugs in the implementation of ResNet in Pytorch. In the valid process, once ResNet was switched to eval mode, the loss leaps. The current workaround is to remain the image encoder module in train mode, no matter in valid or test process.  
- The loss in the text generation task (`text` / `knowledge_*`) may leap after several batches. The current workaround is to kill the running training procedure manually and retrain it from the checkpoint (use the same `<model_file>`). The final loss for the `text`, `knowledge_styletip`, `knowledge_attribute`, `knowledge_celebrity` task is about 3.6, 3.4, 2.6, 2.7 respectively. 
