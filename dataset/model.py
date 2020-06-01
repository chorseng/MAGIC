"""Data models.

Data models:
    * Utterance
    * Product
    * TidyUtterance

"""

from typing import List, Dict, Any

from config import DatasetConfig
from util import pad_or_clip_text, pad_or_clip_images


class Utterance():
    """Utterance data model.

    Attributes:
        speaker (int): Speaker.
                       0 (USER_SPEAKER) for user, 1 (SYS_SPEAKER) for system.
        utter_type (int): Utterance type.
        text (str): Text.
        pos_images (List[int]): Positive images.
        neg_images (List[int]): Negative images.

    """

    def __init__(self, speaker: int, utter_type: int, text: List[int],
                 pos_images: List[int], neg_images: List[int]):
        self.speaker: int = speaker
        self.utter_type: int = utter_type
        self.text: List[int] = text
        self.pos_images: List[int] = pos_images
        self.neg_images: List[int] = neg_images

    def __repr__(self):
        return str((self.speaker, self.utter_type, self.text,
                    self.pos_images, self.neg_images))


class Product():
    """Product data model.

    Attributes:
        product_name (str): Product name, which is the name of the .json file.
        attribute_dict (Dict[str, Any]): Attribute dictionary.

    """

    def __init__(self, product_name: str, attribute_dict: Dict[str, Any]):
        self.product_name = product_name
        self.attribute_dict = attribute_dict


class TidyUtterance():
    """Tidy utterance data model.

    Attributes:
        utter_type (int): Utterance type.
        text (List[int]): Text.
        text_len (int): Text length.
        pos_images (List[int]): Positive images.
        pos_images_num (int): Number of positive images.
        neg_images (List[int]): Negative images.
        neg_images_num (int): Number of negative images.

    """

    def __init__(self, utter: Utterance):
        self.utter_type: int = utter.utter_type
        self.text, self.text_len = pad_or_clip_text(
            utter.text, DatasetConfig.dialog_text_max_len)
        self.pos_images, self.pos_images_num = pad_or_clip_images(
            utter.pos_images, DatasetConfig.pos_images_max_num)
        self.neg_images, self.neg_images_num = pad_or_clip_images(
            utter.neg_images, DatasetConfig.neg_images_max_num)

    def __repr__(self):
        return str((self.utter_type,
                    self.text, self.text_len,
                    self.pos_images, self.pos_images_num,
                    self.neg_images, self.neg_images_num))
