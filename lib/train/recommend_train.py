from datetime import datetime
from itertools import chain
from os.path import isfile

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import RecommendTrainConfig, GlobalConfig
from config.model_config import SimilarityConfig
from constant import RECOMMEND_TASK
from dataset import Dataset
from lib import encode_context
from lib.loss import recommend_loss
from lib.valid import recommend_valid
from lib.test import recommend_test
from model import Similarity
from model import TextEncoder, ImageEncoder, ContextEncoder


def recommend_train(
        context_text_encoder: TextEncoder,
        context_image_encoder: ImageEncoder,
        context_encoder: ContextEncoder,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
        model_file: str,
        vocab_size: int,
        embed_init=None
):
    """Recommend train.

    Args:
        context_text_encoder (TextEncoder): Context text encoder.
        context_image_encoder (ImageEncoder): Context image encoder.
        context_encoder (ContextEncoder): Context encoder.
        train_dataset (Dataset): Train dataset.
        valid_dataset (Dataset): Valid dataset.
        test_dataset (Dataset): Test dataset.
        model_file (str): Saved model file.
        vocab_size (int): Vocabulary size.
        embed_init: Initial embedding (vocab_size, embed_size).

    """

    # Data loader.
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=RecommendTrainConfig.batch_size,
        shuffle=True,
        num_workers=RecommendTrainConfig.num_data_loader_workers)

    # Model.
    similarity_config = SimilarityConfig(vocab_size, embed_init)
    similarity = Similarity(similarity_config).to(GlobalConfig.device)

    # Model parameters.
    params = list(chain.from_iterable([list(model.parameters()) for model in [
        context_text_encoder,
        context_image_encoder,
        context_encoder,
        similarity
    ]]))

    optimizer = Adam(params, lr=RecommendTrainConfig.learning_rate)
    epoch_id = 0
    min_valid_loss = None

    # Load saved state.
    if isfile(model_file):
        state = torch.load(model_file)
        similarity.load_state_dict(state['similarity'])
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
    similarity.train()

    finished = False

    for epoch_id in range(epoch_id, RecommendTrainConfig.num_iterations):
        for batch_id, train_data in enumerate(train_data_loader):
            # Sets gradients to 0.
            optimizer.zero_grad()

            context_dialog, pos_products, neg_products = train_data

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

            loss = recommend_loss(similarity, batch_size, context,
                                  pos_products, neg_products)
            sum_loss += loss

            loss.backward()
            optimizer.step()

            # Print loss every `TrainConfig.print_freq` batches.
            if (batch_id + 1) % RecommendTrainConfig.print_freq == 0:
                cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sum_loss /= RecommendTrainConfig.print_freq
                print('epoch: {} \tbatch: {} \tloss: {} \ttime: {}'.format(
                    epoch_id + 1, batch_id + 1, sum_loss, cur_time))
                sum_loss = 0

            # Valid every `TrainConfig.valid_freq` batches.
            if (batch_id + 1) % RecommendTrainConfig.valid_freq == 0:
                valid_loss = recommend_valid(context_text_encoder,
                                             context_image_encoder,
                                             context_encoder,
                                             similarity,
                                             valid_dataset)
                cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print('valid_loss: {} \ttime: {}'.format(valid_loss, cur_time))

                # Save current best model.
                if min_valid_loss is None or valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    bad_loss_cnt = 0
                    save_dict = {
                        'task': RECOMMEND_TASK,
                        'epoch_id': epoch_id,
                        'min_valid_loss': min_valid_loss,
                        'optimizer': optimizer.state_dict(),

                        'context_text_encoder':
                            context_text_encoder.state_dict(),
                        'context_image_encoder':
                            context_image_encoder.state_dict(),
                        'context_encoder':
                            context_encoder.state_dict(),
                        'similarity': similarity.state_dict()
                    }
                    torch.save(save_dict, model_file)
                    print('Best model saved.')
                else:
                    bad_loss_cnt += 1

                if bad_loss_cnt > RecommendTrainConfig.patience:
                    recommend_test(context_text_encoder,
                                   context_image_encoder,
                                   context_encoder,
                                   similarity,
                                   test_dataset)
                    finished = True
                    break
        if finished:
            break
