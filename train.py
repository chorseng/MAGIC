#!/usr/bin/python3

"""Train module.

Usage:
    python train.py <task> <model_name>

"""

import argparse
from os.path import isfile, join
from typing import List, Dict, Union

import torch

from config import GlobalConfig
from config.dataset_config import DatasetConfig
from config.model_config import ContextEncoderConfig
from config.model_config import ContextImageEncoderConfig
from config.model_config import ContextTextEncoderConfig
from constant import (INTENTION_TASK, TEXT_TASK, RECOMMEND_TASK,
                      KNOWLEDGE_STYLETIP_SUBTASK,
                      KNOWLEDGE_ATTRIBUTE_SUBTASK, KNOWLEDGE_CELEBRITY_SUBTASK)
from constant import TASK_STR, TASK_ID
from constant import TRAIN_MODE, VALID_MODE, TEST_MODE
from dataset import Dataset
from dataset.raw_data import CommonData
from dataset.tidy_data import TidyDialog
from dataset.knowledge_data import KnowledgeData
from lib.train import intention_train, text_train, recommend_train
from lib.train import (knowledge_styletip_train, knowledge_attribute_train,
                       knowledge_celebrity_train)
from model import TextEncoder, ImageEncoder, ContextEncoder
from util import load_pkl, get_embed_init

# Constants.
TASKS: List[str] = list(TASK_STR.values())


def train(task: int, model_file_name: str):
    """Train model.

    Args:
        task (int): Task.
        model_file_name (str): Model file name (saved or to be saved).

    """

    # Check if data exists.
    if not isfile(DatasetConfig.common_raw_data_file):
        raise ValueError('No common raw data.')

    # Load extracted common data.
    common_data: CommonData = load_pkl(DatasetConfig.common_raw_data_file)

    # Dialog data files.
    train_dialog_data_file = DatasetConfig.get_dialog_filename(task,
                                                               TRAIN_MODE)
    valid_dialog_data_file = DatasetConfig.get_dialog_filename(task,
                                                               VALID_MODE)
    test_dialog_data_file = DatasetConfig.get_dialog_filename(task,
                                                              TEST_MODE)
    if not isfile(train_dialog_data_file):
        raise ValueError('No train dialog data file.')
    if not isfile(valid_dialog_data_file):
        raise ValueError('No valid dialog data file.')

    # Load extracted dialogs.
    train_dialogs: List[TidyDialog] = load_pkl(train_dialog_data_file)
    valid_dialogs: List[TidyDialog] = load_pkl(valid_dialog_data_file)
    test_dialogs: List[TidyDialog] = load_pkl(test_dialog_data_file)

    if task in {KNOWLEDGE_STYLETIP_SUBTASK, KNOWLEDGE_ATTRIBUTE_SUBTASK,
                KNOWLEDGE_CELEBRITY_SUBTASK}:
        knowledge_data = KnowledgeData()

    # Dataset wrap.
    train_dataset = Dataset(
        task, common_data.dialog_vocab,
        common_data.image_paths,
        train_dialogs,
        knowledge_data if task == KNOWLEDGE_ATTRIBUTE_SUBTASK else None)
    valid_dataset = Dataset(
        task, common_data.dialog_vocab,
        common_data.image_paths,
        valid_dialogs,
        knowledge_data if task == KNOWLEDGE_ATTRIBUTE_SUBTASK else None)
    test_dataset = Dataset(
        task, common_data.dialog_vocab,
        common_data.image_paths,
        test_dialogs,
        knowledge_data if task == KNOWLEDGE_ATTRIBUTE_SUBTASK else None)

    print('Train dataset size:', len(train_dataset))
    print('Valid dataset size:', len(valid_dataset))
    print('Test dataset size:', len(test_dataset))

    # Get initial embedding.
    vocab_size = len(common_data.dialog_vocab)
    embed_init = get_embed_init(
        common_data.glove, vocab_size).to(GlobalConfig.device)

    # Context model configurations.
    context_text_encoder_config = ContextTextEncoderConfig(
        vocab_size, embed_init)
    context_image_encoder_config = ContextImageEncoderConfig()
    context_encoder_config = ContextEncoderConfig()

    # Context models.
    context_text_encoder = TextEncoder(context_text_encoder_config)
    context_text_encoder = context_text_encoder.to(GlobalConfig.device)
    context_image_encoder = ImageEncoder(context_image_encoder_config)
    context_image_encoder = context_image_encoder.to(GlobalConfig.device)
    context_encoder = ContextEncoder(context_encoder_config)
    context_encoder = context_encoder.to(GlobalConfig.device)

    # Load model file.
    model_file = join(DatasetConfig.dump_dir, model_file_name)
    if isfile(model_file):
        state = torch.load(model_file)
        # if task != state['task']:
        #     raise ValueError("Task doesn't match.")
        context_text_encoder.load_state_dict(state['context_text_encoder'])
        context_image_encoder.load_state_dict(state['context_image_encoder'])
        context_encoder.load_state_dict(state['context_encoder'])

    # Task-specific parts.
    if task == INTENTION_TASK:
        intention_train(
            context_text_encoder,
            context_image_encoder,
            context_encoder,
            train_dataset,
            valid_dataset,
            test_dataset,
            model_file
        )
    elif task == TEXT_TASK:
        text_train(
            context_text_encoder,
            context_image_encoder,
            context_encoder,
            train_dataset,
            valid_dataset,
            test_dataset,
            model_file,
            common_data.dialog_vocab,
            embed_init
        )
    elif task == RECOMMEND_TASK:
        recommend_train(
            context_text_encoder,
            context_image_encoder,
            context_encoder,
            train_dataset,
            valid_dataset,
            test_dataset,
            model_file,
            vocab_size,
            embed_init
        )
    elif task == KNOWLEDGE_STYLETIP_SUBTASK:
        knowledge_styletip_train(
            context_text_encoder,
            context_image_encoder,
            context_encoder,
            train_dataset,
            valid_dataset,
            test_dataset,
            model_file,
            knowledge_data.styletips_data,
            common_data.dialog_vocab,
            embed_init
        )
    elif task == KNOWLEDGE_ATTRIBUTE_SUBTASK:
        knowledge_attribute_train(
            context_text_encoder,
            context_image_encoder,
            context_encoder,
            train_dataset,
            valid_dataset,
            test_dataset,
            model_file,
            knowledge_data.attribute_data,
            common_data.dialog_vocab,
            embed_init
        )
    elif task == KNOWLEDGE_CELEBRITY_SUBTASK:
        knowledge_celebrity_train(
            context_text_encoder,
            context_image_encoder,
            context_encoder,
            train_dataset,
            valid_dataset,
            test_dataset,
            model_file,
            knowledge_data.celebrity_data,
            common_data.dialog_vocab,
            embed_init
        )


def parse_cmd() -> Dict[str, List[str]]:
    """Parse commandline parameters.

    Returns:
        Dict[str, List[str]]: Parse result.

    """

    # Definition of argument parser.
    parser = argparse.ArgumentParser(description='Train.')
    parser.add_argument(
        'task',
        metavar='<task>',
        choices=TASKS,
        help='task ({})'.format('/'.join(TASKS))
    )
    parser.add_argument('model_file', metavar='<model_file>')

    # Namespace -> Dict
    parse_res: Dict[str, Union[List[str], str]] = vars(parser.parse_args())
    return parse_res


def main():
    # Parse commandline parameters and standardize.
    parse_result: Dict[str, Union[List[str], str]] = parse_cmd()
    task: int = TASK_ID[parse_result['task']]
    model_file: str = parse_result['model_file']
    train(task, model_file)


if __name__ == '__main__':
    main()
