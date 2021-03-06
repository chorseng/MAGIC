from typing import List, Dict

import torch
from torch.utils.data import DataLoader

from config import GlobalConfig, TextTestConfig
from dataset import Dataset
from lib import encode_context
from lib.eval import text_eval
from model import TextEncoder, ImageEncoder, ContextEncoder
from model import ToHidden, TextDecoder


def text_test(
        context_text_encoder: TextEncoder,
        context_image_encoder: ImageEncoder,
        context_encoder: ContextEncoder,
        to_hidden: ToHidden,
        text_decoder: TextDecoder,
        test_dataset: Dataset,
        text_length: int,
        vocab: Dict[str, int]):
    """Text test.

    Args:
        context_text_encoder (TextEncoder): Context text encoder.
        context_image_encoder (ImageEncoder): Context image encoder.
        context_encoder (ContextEncoder): Context encoder.
        to_hidden (ToHidden): Context to hidden.
        text_decoder (TextDecoder): Text decoder.
        test_dataset (Dataset): Valid dataset.
        text_length (int): Text length.
        vocab (Dict[str, int]): Vocabulary.

    """

    id2word: List[str] = [None] * len(vocab)
    for word, wid in vocab.items():
        id2word[wid] = word

    # Test dataset loader.
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=TextTestConfig.batch_size,
        num_workers=TextTestConfig.num_data_loader_workers
    )

    # Switch to eval mode.
    context_text_encoder.eval()
    context_image_encoder.eval()
    context_encoder.eval()
    to_hidden.eval()
    text_decoder.eval()

    output_file = open('text.out', 'w')

    with torch.no_grad():
        for batch_id, valid_data in enumerate(test_data_loader):
            texts, text_lengths, images, utter_types = valid_data
            # Sizes:
            # texts: (batch_size, dialog_context_size + 1, dialog_text_max_len)
            # text_lengths: (batch_size, dialog_context_size + 1)
            # images: (batch_size, dialog_context_size + 1,
            #          pos_images_max_num, 3, image_size, image_size)

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

            text_eval(to_hidden, text_decoder, text_length, id2word, context,
                      texts[-1], hiddens, output_file=output_file)

    output_file.close()