import torch
from torch.utils.data import DataLoader

from config import GlobalConfig, IntentionTestConfig
from dataset import Dataset
from lib import encode_context
from model import Intention
from model import TextEncoder, ImageEncoder, ContextEncoder
from util import nll_loss


def intention_test(
        context_text_encoder: TextEncoder,
        context_image_encoder: ImageEncoder,
        context_encoder: ContextEncoder,
        intention: Intention,
        test_dataset: Dataset):
    """Intention test.

    Args:
        context_text_encoder (TextEncoder): Context text encoder.
        context_image_encoder (ImageEncoder): Context image encoder.
        context_encoder (ContextEncoder): Context encoder.
        intention (Intention): Intention.
        test_dataset (Dataset): Test dataset.

    """

    # Test dataset loader.
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=IntentionTestConfig.batch_size,
        shuffle=False,
        num_workers=IntentionTestConfig.num_data_loader_workers
    )

    sum_accuracy = 0

    # Switch to eval mode.
    context_text_encoder.eval()
    context_image_encoder.eval()
    context_encoder.eval()
    intention.eval()

    with torch.no_grad():
        for batch_id, valid_data in enumerate(test_data_loader):

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

            intentions = torch.argmax(intent_prob, dim=1)
            eqs = torch.eq(intentions, utter_types)
            num_correct = torch.sum(eqs).item()
            accuracy = num_correct * 1.0 / eqs.size(0)
            sum_accuracy += accuracy

            # Print.
            print('pred:', intentions)
            print('true:', utter_types)
            print('# correct:', num_correct)
            print('accuracy:', accuracy)
            print('total accuracy:', sum_accuracy / (batch_id + 1))
