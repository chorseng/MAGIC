from datetime import datetime
from itertools import chain
from os.path import isfile

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import IntentionTrainConfig, GlobalConfig
from config.model_config import IntentionConfig
from constant import INTENTION_TASK
from dataset import Dataset
from lib import encode_context
from lib.valid.intention_valid import intention_valid
from lib.test.intention_test import intention_test
from model import Intention
from model import TextEncoder, ImageEncoder, ContextEncoder
from util import nll_loss


def intention_train(
        context_text_encoder: TextEncoder,
        context_image_encoder: ImageEncoder,
        context_encoder: ContextEncoder,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
        model_file: str
):
    """Intention train.
    Args:
        context_text_encoder (TextEncoder): Context text encoder.
        context_image_encoder (ImageEncoder): Context image encoder.
        context_encoder (ContextEncoder): Context encoder.
        train_dataset (Dataset): Train dataset.
        valid_dataset (Dataset): Valid dataset.
        test_dataset (Dataset): Test dataset.
        model_file (str): Saved model file.
    """

    # Data loader.
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=IntentionTrainConfig.batch_size,
        shuffle=True,
        num_workers=IntentionTrainConfig.num_data_loader_workers)

    # Model.
    intention_config = IntentionConfig()
    intention = Intention(intention_config).to(GlobalConfig.device)

    # Model parameters.
    params = list(chain.from_iterable([list(model.parameters()) for model in [
        context_text_encoder,
        context_image_encoder,
        context_encoder,
        intention
    ]]))

    optimizer = Adam(params, lr=IntentionTrainConfig.learning_rate)
    epoch_id = 0
    min_valid_loss = None

    # Load saved state.
    if isfile(model_file):
        state = torch.load(model_file)
        intention.load_state_dict(state['intention'])
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
    intention.train()

    finished = False

    for epoch_id in range(epoch_id, IntentionTrainConfig.num_iterations):
        for batch_id, train_data in enumerate(train_data_loader):
            # Sets gradients to 0.
            optimizer.zero_grad()

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

            loss.backward()
            optimizer.step()

            # Print loss every `TrainConfig.print_freq` batches.
            if (batch_id + 1) % IntentionTrainConfig.print_freq == 0:
                cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sum_loss /= IntentionTrainConfig.print_freq
                print('epoch: {} \tbatch: {} \tloss: {} \ttime: {}'.format(
                    epoch_id + 1, batch_id + 1, sum_loss, cur_time))
                sum_loss = 0

            # Valid every `TrainConfig.valid_freq` batches.
            if (batch_id + 1) % IntentionTrainConfig.valid_freq == 0:
                valid_loss, accuracy = intention_valid(context_text_encoder,
                                                       context_image_encoder,
                                                       context_encoder,
                                                       intention,
                                                       valid_dataset)
                cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print('valid_loss: {} \taccuracy: {} \ttime: {}'.format(
                    valid_loss, accuracy, cur_time))

                # Save current best model.
                if min_valid_loss is None or valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    bad_loss_cnt = 0
                    save_dict = {
                        'task': INTENTION_TASK,
                        'epoch_id': epoch_id,
                        'min_valid_loss': min_valid_loss,
                        'optimizer': optimizer.state_dict(),

                        'context_text_encoder':
                            context_text_encoder.state_dict(),
                        'context_image_encoder':
                            context_image_encoder.state_dict(),
                        'context_encoder':
                            context_encoder.state_dict(),
                        'intention': intention.state_dict()
                    }
                    torch.save(save_dict, model_file)
                    print('Best model saved.')
                else:
                    bad_loss_cnt += 1
                    if bad_loss_cnt > IntentionTrainConfig.patience:
                        intention_test(context_text_encoder,
                                       context_image_encoder,
                                       context_encoder,
                                       intention,
                                       test_dataset)
                        finished = True
                        break
        if finished:
            break
