from datetime import datetime
from functools import partial
from itertools import chain
from os.path import isfile
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import KnowledgeAttributeTrainConfig, GlobalConfig
from config.model_config import (MemoryConfig,
                                 AttributeKVMemoryConfig,
                                 KnowledgeTextDecoderConfig)
from constant import KNOWLEDGE_ATTRIBUTE_SUBTASK
from dataset import Dataset
from dataset.knowledge_data import AttributeData
from lib import encode_context
from lib.loss import text_loss
from lib.test import knowledge_attribute_test
from lib.valid import knowledge_attribute_valid
from model import TextDecoder, ToHidden, KVMemory
from model import TextEncoder, ImageEncoder, ContextEncoder


def knowledge_attribute_train(
        context_text_encoder: TextEncoder,
        context_image_encoder: ImageEncoder,
        context_encoder: ContextEncoder,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
        model_file: str,
        attribute_data: AttributeData,
        vocab: Dict[str, int],
        embed_init=None
):
    """Knowledge styletip train.

    Args:
        context_text_encoder (TextEncoder): Context text encoder.
        context_image_encoder (ImageEncoder): Context image encoder.
        context_encoder (ContextEncoder): Context encoder.
        train_dataset (Dataset): Train dataset.
        valid_dataset (Dataset): Valid dataset.
        test_dataset (Dataset): Test dataset.
        model_file (str): Saved model file.
        attribute_data (AttributeData): Attribute data.
        vocab (Dict[str, int]): Vocabulary.
        embed_init: Initial embedding (vocab_size, embed_size).

    """

    # Data loader.
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=KnowledgeAttributeTrainConfig.batch_size,
        shuffle=True,
        num_workers=KnowledgeAttributeTrainConfig.num_data_loader_workers)

    # Model.
    vocab_size = len(vocab)

    attribute_kv_memory_config = AttributeKVMemoryConfig(
        len(attribute_data.key_vocab),
        len(attribute_data.value_vocab))
    text_decoder_config = KnowledgeTextDecoderConfig(vocab_size,
                                                     MemoryConfig.memory_size,
                                                     MemoryConfig.output_size,
                                                     embed_init)

    to_hidden = ToHidden(text_decoder_config)
    to_hidden = to_hidden.to(GlobalConfig.device)

    attribute_kv_memory = KVMemory(attribute_kv_memory_config)
    attribute_kv_memory = attribute_kv_memory.to(GlobalConfig.device)

    text_decoder = TextDecoder(text_decoder_config)
    text_decoder = text_decoder.to(GlobalConfig.device)

    # Model parameters.
    params = list(chain.from_iterable([list(model.parameters()) for model in [
        context_text_encoder,
        context_image_encoder,
        context_encoder,
        to_hidden,
        attribute_kv_memory,
        text_decoder
    ]]))

    optimizer = Adam(params, lr=KnowledgeAttributeTrainConfig.learning_rate)
    epoch_id = 0
    min_valid_loss = None

    # Load saved state.
    if isfile(model_file):
        state = torch.load(model_file)
        to_hidden.load_state_dict(state['to_hidden'])
        attribute_kv_memory.load_state_dict(state['attribute_kv_memory'])
        text_decoder.load_state_dict(state['text_decoder'])
        optimizer.load_state_dict(state['optimizer'])
        epoch_id = state['epoch_id']
        min_valid_loss = state['min_valid_loss']

    # Loss.
    sum_loss = 0
    bad_loss_cnt = 0

    # Switch to train mode.
    context_text_encoder.train()
    context_image_encoder.train()
    context_encoder.train()
    to_hidden.train()
    attribute_kv_memory.train()
    text_decoder.train()

    finished = False

    for epoch_id in range(epoch_id,
                          KnowledgeAttributeTrainConfig.num_iterations):
        for batch_id, train_data in enumerate(train_data_loader):
            # Set gradients to 0.
            optimizer.zero_grad()

            train_data, products = train_data
            keys, values, pair_length = products
            keys = keys.to(GlobalConfig.device)
            values = values.to(GlobalConfig.device)
            pair_length = pair_length.to(GlobalConfig.device)

            texts, text_lengths, images, utter_types = train_data
            # Sizes:
            # texts: (batch_size, dialog_context_size + 1, dialog_text_max_len)
            # text_lengths: (batch_size, dialog_context_size + 1)
            # images: (batch_size, dialog_context_size + 1,
            #          pos_images_max_num, 3, image_size, image_size)
            # utter_types: (batch_size, )

            # To device.

            texts = texts.to(GlobalConfig.device)
            text_lengths = text_lengths.to(GlobalConfig.device)
            images = images.to(GlobalConfig.device)
            utter_types = utter_types.to(GlobalConfig.device)

            texts.transpose_(0, 1)
            # (dialog_context_size + 1, batch_size, dialog_text_max_len)

            text_lengths.transpose_(0, 1)
            # (dialog_context_size + 1, batch_size)

            images.transpose_(0, 1)
            images.transpose_(1, 2)
            # (dialog_context_size + 1, pos_images_max_num, batch_size, 3,
            #  image_size, image_size)

            # Encode context.
            context, hiddens = encode_context(
                context_text_encoder,
                context_image_encoder,
                context_encoder,
                texts,
                text_lengths,
                images
            )
            # (batch_size, context_vector_size)

            encode_knowledge_func = partial(attribute_kv_memory,
                                            keys, values, pair_length)

            loss, n_totals = text_loss(to_hidden, text_decoder,
                                       text_decoder_config.text_length, context,
                                       texts[-1], text_lengths[-1],
                                       hiddens, encode_knowledge_func)
            sum_loss += loss / text_decoder_config.text_length

            loss.backward()
            optimizer.step()

            # Print loss every `TrainConfig.print_freq` batches.
            if (batch_id + 1) % KnowledgeAttributeTrainConfig.print_freq == 0:
                cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sum_loss /= KnowledgeAttributeTrainConfig.print_freq
                print('epoch: {} \tbatch: {} \tloss: {} \ttime: {}'.format(
                    epoch_id + 1, batch_id + 1, sum_loss, cur_time))
                sum_loss = 0

            # Valid every `TrainConfig.valid_freq` batches.
            if (batch_id + 1) % KnowledgeAttributeTrainConfig.valid_freq == 0:
                valid_loss = knowledge_attribute_valid(
                    context_text_encoder,
                    context_image_encoder,
                    context_encoder,
                    to_hidden,
                    attribute_kv_memory,
                    text_decoder,
                    valid_dataset,
                    text_decoder_config.text_length)
                cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print('valid_loss: {} \ttime: {}'.format(valid_loss, cur_time))

                # Save current best model.
                if min_valid_loss is None or valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    bad_loss_cnt = 0
                    save_dict = {
                        'task': KNOWLEDGE_ATTRIBUTE_SUBTASK,
                        'epoch_id': epoch_id,
                        'min_valid_loss': min_valid_loss,
                        'optimizer': optimizer.state_dict(),

                        'context_text_encoder':
                            context_text_encoder.state_dict(),
                        'context_image_encoder':
                            context_image_encoder.state_dict(),
                        'context_encoder':
                            context_encoder.state_dict(),
                        'to_hidden':
                            to_hidden.state_dict(),
                        'attribute_kv_memory':
                            attribute_kv_memory.state_dict(),
                        'text_decoder':
                            text_decoder.state_dict()
                    }
                    torch.save(save_dict, model_file)
                    print('Best model saved.')
                else:
                    bad_loss_cnt += 1
                    if bad_loss_cnt > KnowledgeAttributeTrainConfig.patience:
                        knowledge_attribute_test(context_text_encoder,
                                                 context_image_encoder,
                                                 context_encoder,
                                                 to_hidden,
                                                 attribute_kv_memory,
                                                 text_decoder,
                                                 test_dataset,
                                                 text_decoder_config.text_length,
                                                 vocab)
                        finished = True
                        break
        if finished:
            break
