from functools import partial

import torch
from torch.utils.data import DataLoader

from config import GlobalConfig, KnowledgeAttributeValidConfig
from dataset import Dataset
from lib import encode_context
from lib.loss import text_loss
from model import TextEncoder, ImageEncoder, ContextEncoder
from model import ToHidden, TextDecoder, KVMemory


def knowledge_attribute_valid(
        context_text_encoder: TextEncoder,
        context_image_encoder: ImageEncoder,
        context_encoder: ContextEncoder,
        to_hidden: ToHidden,
        attribute_kv_memory: KVMemory,
        text_decoder: TextDecoder,
        valid_dataset: Dataset,
        text_length: int):
    """Knowledge attribute valid.

    Args:
        context_text_encoder (TextEncoder): Context text encoder.
        context_image_encoder (ImageEncoder): Context image encoder.
        context_encoder (ContextEncoder): Context encoder.
        to_hidden (ToHidden): Context to hidden.
        attribute_kv_memory (KVMemory): Attribute Key-Value Memory.
        text_decoder (TextDecoder): Text decoder.
        valid_dataset (Dataset): Valid dataset.
        text_length (int): Text length.

    """

    # Valid dataset loader.
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=KnowledgeAttributeValidConfig.batch_size,
        shuffle=True,
        num_workers=KnowledgeAttributeValidConfig.num_data_loader_workers
    )

    sum_loss = 0
    num_batches = 0

    # Switch to eval mode.
    context_text_encoder.eval()
    context_image_encoder.eval()
    context_encoder.eval()
    to_hidden.eval()
    attribute_kv_memory.eval()
    text_decoder.eval()

    with torch.no_grad():
        for batch_id, valid_data in enumerate(valid_data_loader):
            # Only valid `ValidConfig.num_batches` batches.
            if batch_id >= KnowledgeAttributeValidConfig.num_batches:
                break

            num_batches += 1

            valid_data, products = valid_data
            keys, values, pair_length = products
            keys = keys.to(GlobalConfig.device)
            values = values.to(GlobalConfig.device)
            pair_length = pair_length.to(GlobalConfig.device)

            texts, text_lengths, images, utter_types = valid_data

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

            loss, n_totals = text_loss(to_hidden, text_decoder, text_length,
                                       context, texts[-1], text_lengths[-1],
                                       hiddens, encode_knowledge_func)
            sum_loss += loss / text_length

    # Switch to train mode.
    context_text_encoder.train()
    context_image_encoder.train()
    context_encoder.train()
    to_hidden.train()
    attribute_kv_memory.train()
    text_decoder.train()

    return sum_loss / num_batches
