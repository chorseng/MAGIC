from functools import partial
from typing import List, Dict

import torch
from torch.utils.data import DataLoader

from config import GlobalConfig, KnowledgeCelebrityTestConfig
from dataset import Dataset
from lib import encode_context
from lib.eval import text_eval
from model import TextEncoder, ImageEncoder, ContextEncoder
from model import ToHidden, TextDecoder, Memory


def knowledge_celebrity_test(
        context_text_encoder: TextEncoder,
        context_image_encoder: ImageEncoder,
        context_encoder: ContextEncoder,
        to_hidden: ToHidden,
        celebrity_memory: Memory,
        text_decoder: TextDecoder,
        test_dataset: Dataset,
        celebrity_scores,
        text_length: int,
        vocab: Dict[str, int]):
    """Knowledge celebrity test.

    Args:
        context_text_encoder (TextEncoder): Context text encoder.
        context_image_encoder (ImageEncoder): Context image encoder.
        context_encoder (ContextEncoder): Context encoder.
        to_hidden (ToHidden): Context to hidden.
        celebrity_memory (Memory): Celebrity Memory.
        text_decoder (TextDecoder): Text decoder.
        test_dataset (Dataset): Valid dataset.
        celebrity_scores: Celebrity scores.
        text_length (int): Text length.
        vocab (Dict[str, int]): Vocabulary.

    """

    id2word: List[str] = [None] * len(vocab)
    for word, wid in vocab.items():
        id2word[wid] = word

    # Test dataset loader.
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=KnowledgeCelebrityTestConfig.batch_size,
        num_workers=KnowledgeCelebrityTestConfig.num_data_loader_workers
    )

    sum_loss = 0

    # Switch to eval mode.
    context_text_encoder.eval()
    context_image_encoder.eval()
    context_encoder.eval()
    to_hidden.eval()
    celebrity_memory.eval()
    text_decoder.eval()

    output_file = open('knowledge_celebrity.out', 'w')

    with torch.no_grad():
        for batch_id, test_data in enumerate(test_data_loader):
            texts, text_lengths, images, utter_types = test_data
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

            text_eval(to_hidden, text_decoder, text_length, id2word,
                      context, texts[-1], hiddens,
                      encode_knowledge_func, output_file=output_file)

    output_file.close()