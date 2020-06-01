import torch

from config import DatasetConfig, GlobalConfig
from constant import SOS_ID
from model import TextEncoder, ImageEncoder, ContextEncoder


def encode_context(context_text_encoder: TextEncoder,
                   context_image_encoder: ImageEncoder,
                   context_encoder: ContextEncoder,
                   texts, text_lengths, images):
    """ Encode context.

    Args:
        context_text_encoder (TextEncoder): Context text encoder.
        context_image_encoder (ImageEncoder): Context image encoder.
        context_encoder (ContextEncoder): Context encoder.
        texts: Texts (dialog_context_size + 1, batch_size, dialog_text_max_len)
        text_lengths: Text lengths (dialog_context_size + 1, batch_size)
        images: Images (dialog_context_size + 1, pos_images_max_num,
                        batch_size, 3, image_size, image_size)

    Returns:
        Context vector (batch_size, context_vector_size)
          context_vector_size = hidden_size * num_layers * num_directions

    """
    batch_size = texts.size(1)

    sos = SOS_ID * torch.ones(batch_size, dtype=torch.long).view(-1, 1)
    sos = sos.to(GlobalConfig.device)
    # (batch_size, 1)

    context = []
    hiddens = None
    for i in range(DatasetConfig.dialog_context_size):
        text, text_length = texts[i], text_lengths[i]
        text = text.to(GlobalConfig.device)
        text_length = text_length.to(GlobalConfig.device)
        # text: (batch_size, dialog_text_max_len)
        # text_length: (batch_size, )

        # Insert SOS (Start Of Sentence) to the start of sentence
        text = torch.cat((sos, text), 1).to(GlobalConfig.device)
        text_length += 1

        encoded_text, hiddens = context_text_encoder(text, text_length)
        encoded_text = encoded_text.to(GlobalConfig.device)
        # (batch_size, text_feat_size)
        # text_feat_size = hidden_size * num_layers * num_directions

        for j in range(DatasetConfig.pos_images_max_num):
            image = images[i][j]
            # (batch_size, 3, image_size, image_size)

            encoded_image = context_image_encoder(image, encoded_text)
            encoded_image = encoded_image.to(GlobalConfig.device)
            # (batch_size, )

            mm = torch.cat((encoded_text, encoded_image), 1)
            mm = mm.to(GlobalConfig.device)
            # (batch_size, text_feat_size + image_feat_size)

            context.append(mm)

    context = torch.stack(context)
    context = context.to(GlobalConfig.device)
    # (dialog_context_size, batch_size, text_feat_size + image_feat_size)

    context = context_encoder(context)
    # (batch_size, context_vector_size)
    # context_vector_size = hidden_size * num_layers * num_directions

    return context, hiddens
