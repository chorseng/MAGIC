import torch
from torch.utils.data import DataLoader

from config import GlobalConfig, DatasetConfig, RecommendTestConfig
from dataset import Dataset
from lib import encode_context
from lib.loss import recommend_loss
from model import Similarity
from model import TextEncoder, ImageEncoder, ContextEncoder
from lib.eval import recommend_eval


def recommend_test(
        context_text_encoder: TextEncoder,
        context_image_encoder: ImageEncoder,
        context_encoder: ContextEncoder,
        similarity: Similarity,
        test_dataset: Dataset):
    """Recommend test.

    Args:
        context_text_encoder (TextEncoder): Context text encoder.
        context_image_encoder (ImageEncoder): Context image encoder.
        context_encoder (ContextEncoder): Context encoder.
        similarity (Similarity): Intention.
        test_dataset (Dataset): Valid dataset.

    """

    # Test dataset loader.
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=RecommendTestConfig.batch_size,
        num_workers=RecommendTestConfig.num_data_loader_workers
    )

    # Switch to eval mode.
    context_text_encoder.eval()
    context_image_encoder.eval()
    context_encoder.eval()
    # similarity.eval()
    # There might be a bug in the implement of resnet.

    num_ranks = torch.zeros(DatasetConfig.neg_images_max_num + 1,
                            dtype=torch.long)
    num_ranks = num_ranks.to(GlobalConfig.device)
    total_samples = 0

    with torch.no_grad():
        for batch_id, valid_data in enumerate(test_data_loader):
            # Only valid `ValidConfig.num_batches` batches.

            context_dialog, pos_products, neg_products = valid_data

            texts, text_lengths, images, utter_types = context_dialog
            # Sizes:
            # texts: (batch_size, dialog_context_size + 1, dialog_text_max_len)
            # text_lengths: (batch_size, dialog_context_size + 1)
            # images: (batch_size, dialog_context_size + 1,
            #          pos_images_max_num, 3, image_size, image_size)
            # utter_types: (batch_size, )

            batch_size = texts.size(0)

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
            context, _ = encode_context(
                context_text_encoder,
                context_image_encoder,
                context_encoder,
                texts,
                text_lengths,
                images
            )
            # (batch_size, context_vector_size)

            num_rank = recommend_eval(
                similarity,
                batch_size,
                context,
                pos_products,
                neg_products
            )

            total_samples += batch_size
            num_ranks += num_rank

            # TODO: set freq.
            for i in range(DatasetConfig.neg_images_max_num):
                print('total recall@{} = {}'.format(
                    i + 1, torch.sum(num_ranks[:i + 1]).item() / total_samples))
