"""Dataset module."""
import json
from os.path import join, isfile
from typing import List, Dict

import torch
from PIL import Image
from nltk import word_tokenize
from torch.utils import data

from config import DatasetConfig
from constant import INTENTION_TASK, TEXT_TASK, RECOMMEND_TASK
from constant import KNOWLEDGE_ATTRIBUTE_SUBTASK
from constant import KNOWLEDGE_CELEBRITY_SUBTASK
from constant import KNOWLEDGE_STYLETIP_SUBTASK
from constant import UNK_ID, PAD_ID, EOS_ID
from dataset.tidy_data import TidyDialog
from util import pad_or_clip_text, get_product_path
from dataset import KnowledgeData


class Dataset(data.Dataset):
    """Dataset class."""

    # Constants.
    EMPTY_IMAGE = torch.zeros(3, DatasetConfig.image_size,
                              DatasetConfig.image_size)
    EMPTY_PRODUCT_TEXT = [EOS_ID] + [PAD_ID] * (
            DatasetConfig.product_text_max_len - 1)

    def __init__(self, task: int,
                 dialog_vocab: Dict[str, int],
                 image_paths: List[str],
                 dialogs: List[TidyDialog],
                 knowledge_data: KnowledgeData = None):
        self.task: int = task
        self.dialog_vocab: Dict[str, int] = dialog_vocab
        self.image_paths: List[str] = image_paths
        self.dialogs: List[TidyDialog] = dialogs

        if knowledge_data is not None:
            self.knowledge_data = knowledge_data
            self.key_vocab = self.knowledge_data.attribute_data.key_vocab
            self.value_vocab = self.knowledge_data.attribute_data.value_vocab

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, index: int):
        """ Get item for a given index.

        Args:
            index (int): item index.

        Returns:
            - INTENTION_TASK, TEXT_TASK, KNOWLEDGE_STYLETIP_SUBTASK and
              KNOWLEDGE_CELEBRITY_SUBTASK
                texts: Texts (dialog_context_size + 1, dialog_text_max_len).
                text_lengths: Text lengths (dialog_context_size + 1, ).
                images: Images (dialog_context_size + 1, pos_images_max_num, 3,
                               image_size, image_size).
                utter_type (int): The type of the last user utterance.

            - RECOMMEND_TASK
                context_dialog:
                    texts: Texts (dialog_context_size + 1, dialog_text_max_len).
                    text_lengths: Text lengths (dialog_context_size + 1, ).
                    images: Images (dialog_context_size + 1, pos_images_max_num,
                                    3, image_size, image_size).
                    utter_type (int): The type of the last user utterance.
                pos_products:
                    num_pos_products (int): Number of positive products.
                    pos_images: Positive images
                                (pos_images_max_num, 3, image_size, image_size).
                    pos_product_texts: Positive product texts
                                     (pos_images_max_num, product_text_max_len).
                    pos_product_text_lengths: Positive product text lengths
                                          (pos_images_max_num, ).
                neg_products:
                    num_neg_products (int): Number of negative products.
                    neg_images: Negative images
                                (neg_images_max_num, 3, image_size, image_size).
                    neg_product_texts: Negative product texts
                                     (neg_images_max_num, product_text_max_len).
                    neg_product_text_lengths: Negative product text lengths
                                          (neg_images_max_num, ).

        """

        dialog: TidyDialog = self.dialogs[index % len(self.dialogs)]
        if self.task in {INTENTION_TASK,
                         TEXT_TASK,
                         KNOWLEDGE_STYLETIP_SUBTASK,
                         KNOWLEDGE_CELEBRITY_SUBTASK}:
            return self._get_context_dialog(dialog)
        elif self.task == RECOMMEND_TASK:
            context_dialog = self._get_context_dialog(dialog)
            # Products (Text & Image).
            utter = dialog[-1]  # System response.
            pos_products = self._get_images_product_texts(
                utter.pos_images, DatasetConfig.pos_images_max_num)
            neg_products = self._get_images_product_texts(
                utter.neg_images, DatasetConfig.neg_images_max_num)
            return context_dialog, pos_products, neg_products
        elif self.task == KNOWLEDGE_ATTRIBUTE_SUBTASK:
            context_dialog = self._get_context_dialog(dialog)
            attributes = self.get_attributes(dialog[-1].pos_images[0])
            return context_dialog, attributes

        raise ValueError('Invalid task.')

    def _get_context_dialog(self, dialog: TidyDialog):
        """Get context dialog.

        Note: The last utterance of the context dialog is system response.

        Args:
            dialog (TidyDialog): Dialog.

        Returns:
            texts: Texts (dialog_context_size + 1, dialog_text_max_len).
            text_lengths: Text lengths (dialog_context_size + 1, ).
            images: Images (dialog_context_size + 1, pos_images_max_num, 3,
                           image_size, image_size).
            utter_type (int): The type of the last user utterance.

        """
        # Text.
        text_list: List[List[int]] = [utter.text for utter in dialog]
        text_length_list: List[int] = [utter.text_len for utter in dialog]

        # Text tensors.
        texts = torch.stack(tuple([torch.tensor(text) for text in text_list]))
        # (dialog_context_size + 1, dialog_text_max_len)
        text_lengths = torch.tensor(text_length_list)
        # (dialog_context_size + 1, )

        # Image.
        image_list = [[] for _ in range(DatasetConfig.dialog_context_size + 1)]

        for idx, utter in enumerate(dialog):
            for img_id in utter.pos_images:
                path = self.image_paths[img_id]
                if path:
                    path = join(DatasetConfig.image_data_directory, path)
                else:
                    path = ''
                if path and isfile(path):
                    try:
                        raw_image = Image.open(path).convert("RGB")
                        image = DatasetConfig.transform(raw_image)
                        image_list[idx].append(image)
                    except OSError:
                        image_list[idx].append(Dataset.EMPTY_IMAGE)
                else:
                    image_list[idx].append(Dataset.EMPTY_IMAGE)

        images = torch.stack(list(map(torch.stack, image_list)))
        # (dialog_context_size + 1, pos_images_max_num,
        # 3, image_size, image_size)

        # Utterance type.
        utter_type = dialog[-2].utter_type
        return texts, text_lengths, images, utter_type

    def _get_images_product_texts(self, image_ids: List[int],
                                  num_products: int):
        """Get images and product texts of a response.

        Args:
            image_ids (List[int]): Image ids.
            num_products (int): Number of images (max images).

        Returns:
            num_products (int): Number of products (exclude padding).
            images: Images (num_products, 3, image_size, image_size).
            product_texts: Product texts (num_products, product_text_max_len).
            product_text_lengths: Product text lengths (num_products, ).

        """
        images = []
        product_texts = []
        product_text_lengths = []
        for img_id in image_ids:
            if img_id == 0:
                break
            image_name = self.image_paths[img_id]
            image_path = join(DatasetConfig.image_data_directory, image_name)
            product_path = get_product_path(image_name)

            # Image.
            raw_image = Image.open(image_path).convert("RGB")
            image = DatasetConfig.transform(raw_image)
            images.append(image)

            # Text.
            text = Dataset._get_product_text(product_path)
            text = [self.dialog_vocab.get(word, UNK_ID) for word in
                    word_tokenize(text)]
            text, text_len = pad_or_clip_text(
                text, DatasetConfig.product_text_max_len)
            product_texts.append(text)
            product_text_lengths.append(text_len)

        # Padding.
        num_pads = (num_products - len(images))
        images.extend([self.EMPTY_IMAGE] * num_pads)
        product_texts.extend([self.EMPTY_PRODUCT_TEXT] * num_pads)
        product_text_lengths.extend([1] * num_pads)

        # To tensors.
        num_products = len(images)
        images = torch.stack(images)
        product_texts = torch.stack(list(map(torch.tensor, product_texts)))
        product_text_lengths = torch.tensor(product_text_lengths)
        return num_products, images, product_texts, product_text_lengths

    @staticmethod
    def _get_product_text(product_path):
        product_dict = json.load(open(product_path))
        texts = []
        for key, value in product_dict.items():
            # Note: Only a space is also empty.
            if value is not None and value != '' and value != ' ':
                texts.extend([key, value])
        return ' '.join(texts).lower()

    def get_attributes(self, product_id):
        keys = []
        values = []
        if product_id != 0:
            image_name = self.image_paths[product_id]
            product_path = get_product_path(image_name)
            if isfile(product_path):
                product_dict = json.load(open(product_path))
                for key, value in product_dict.items():
                    # Note: Only a space is also empty.
                    if value is not None and value != '' and value != ' ':
                        key = key.lower()
                        value = value.lower()
                        if key not in self.key_vocab:
                            continue
                        key_id = self.key_vocab[key]
                        if (key_id, value) in self.value_vocab:
                            value_id = self.value_vocab[(key_id, value)]
                            keys.append(key_id)
                            values.append(value_id)
        length = len(keys)
        pad = [0] * (len(self.key_vocab) - length)
        keys.extend(pad)
        values.extend(pad)
        return torch.tensor(keys), torch.tensor(values), torch.tensor(length)
