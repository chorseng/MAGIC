from functools import partial

import torch
from torch.utils.data import DataLoader

from config import GlobalConfig, KnowledgeCelebrityValidConfig
from dataset import Dataset
from lib import encode_context
from lib.loss import text_loss
from model import TextEncoder, ImageEncoder, ContextEncoder
from model import ToHidden, TextDecoder, GraphEncoder, Memory


def knowledge_celebrity_valid(
        context_text_encoder: TextEncoder,
        context_image_encoder: ImageEncoder,
        context_encoder: ContextEncoder,
        to_hidden: ToHidden,
        celebrity_memory: Memory,
        text_decoder: TextDecoder,
        valid_dataset: Dataset,
        celebrity_scores,
        text_length: int):
    """Knowledge celebrity valid.

    Args:
        context_text_encoder (TextEncoder): Context text encoder.
        context_image_encoder (ImageEncoder): Context image encoder.
        context_encoder (ContextEncoder): Context encoder.
        to_hidden (ToHidden): Context to hidden.
        celebrity_memory (Memory): Celebrity Memory.
        text_decoder (TextDecoder): Text decoder.
        valid_dataset (Dataset): Valid dataset.
        celebrity_scores: Celebrity scores.
        text_length (int): Text length.

    """

    # Valid dataset loader.
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=KnowledgeCelebrityValidConfig.batch_size,
        shuffle=True,
        num_workers=KnowledgeCelebrityValidConfig.num_data_loader_workers
    )

    sum_loss = 0
    num_batches = 0

    # Switch to eval mode.
    context_text_encoder.eval()
    context_image_encoder.eval()
    context_encoder.eval()
    to_hidden.eval()
    celebrity_memory.eval()
    text_decoder.eval()

    with torch.no_grad():
        for batch_id, valid_data in enumerate(valid_data_loader):
            # Only valid `ValidConfig.num_batches` batches.
            if batch_id >= KnowledgeCelebrityValidConfig.num_batches:
                break

            num_batches += 1

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
            # utter_types = utter_types.to(GlobalConfig.device)

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

            knowledge_entry = celebrity_scores
            encode_knowledge_func = partial(celebrity_memory, knowledge_entry)

            loss, n_totals = text_loss(to_hidden, text_decoder, text_length,
                                       context, texts[-1], text_lengths[-1],
                                       hiddens, encode_knowledge_func)
            sum_loss += loss / text_length

    # Switch to train mode.
    context_text_encoder.train()
    context_image_encoder.train()
    context_encoder.train()
    to_hidden.train()
    celebrity_memory.train()
    text_decoder.train()

    return sum_loss / num_batches
