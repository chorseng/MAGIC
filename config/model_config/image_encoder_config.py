from config.model_config import ContextTextEncoderConfig
from util.better_abc import ABCMeta, abstract_attribute


class ImageEncoderConfig(metaclass=ABCMeta):
    image_size = abstract_attribute()
    num_channels = abstract_attribute()
    text_feat_size = abstract_attribute()
    image_feat_size = abstract_attribute()


class ContextImageEncoderConfig(ImageEncoderConfig):
    image_size = 1
    num_channels = 512
    text_feat_size = ContextTextEncoderConfig.text_feat_size
    image_feat_size = num_channels


class ProductImageEncoderConfig(ImageEncoderConfig):
    image_size = 1
    num_channels = 512
    text_feat_size = ContextTextEncoderConfig.text_feat_size
    image_feat_size = num_channels
