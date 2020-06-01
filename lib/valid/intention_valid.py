import torch
from torch.utils.data import DataLoader

from config import GlobalConfig, DatasetConfig, IntentionValidConfig
from dataset import Dataset
from lib import encode_context
from model import Intention
from model import TextEncoder, ImageEncoder, ContextEncoder
from util import nll_loss


def intention_valid(
        context_text_encoder: TextEncoder,
        context_image_encoder: ImageEncoder,
        context_encoder: ContextEncoder,
        intention: Intention,
        valid_dataset: Dataset):
    """Intention valid.

    Args:
        context_text_encoder (TextEncoder): Context text encoder.
        context_image_encoder (ImageEncoder): Context image encoder.
        context_encoder (ContextEncoder): Context encoder.
        intention (Intention): Intention.
        valid_dataset (Dataset): Valid dataset.

    """

    # Valid dataset loader.
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=IntentionValidConfig.batch_size,
        shuffle=True,
        num_workers=IntentionValidConfig.num_data_loader_workers
    )

    sum_loss = 0
    sum_accuracy = 0
    num_batches = 0

    # Switch to eval mode.
    context_text_encoder.eval()
    context_image_encoder.eval()
    context_encoder.eval()
    intention.eval()

    with torch.no_grad():
        for batch_id, valid_data in enumerate(valid_data_loader):
            # Only valid `ValidConfig.num_batches` batches.
            if batch_id >= IntentionValidConfig.num_batches:
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
            context, _ = encode_context(
                context_text_encoder,
                context_image_encoder,
                context_encoder,
                texts,
                text_lengths,
                images
            )
            # (batch_size, context_vector_size)

            intent_prob = intention(context)
            # (batch_size, utterance_type_size)

            loss = nll_loss(intent_prob, utter_types)
            sum_loss += loss

            eqs = torch.eq(torch.argmax(intent_prob, dim=1), utter_types)
            accuracy = torch.sum(eqs).item() * 1.0 / eqs.size(0)
            sum_accuracy += accuracy

    # Switch to train mode.
    context_text_encoder.train()
    context_image_encoder.train()
    context_encoder.train()
    intention.train()

    return sum_loss / num_batches, sum_accuracy / num_batches
