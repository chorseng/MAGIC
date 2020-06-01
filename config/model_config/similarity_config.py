import operator

from config.model_config import ContextEncoderConfig
from config.model_config import ProductImageEncoderConfig
from config.model_config import ProductTextEncoderConfig


class SimilarityConfig:
    product_image_encoder_config = ProductImageEncoderConfig()
    mm_size = operator.add(product_image_encoder_config.text_feat_size,
                           product_image_encoder_config.image_feat_size)
    context_vector_size = ContextEncoderConfig.output_size

    def __init__(self, vocab_size, embed_init=None):
        self.product_text_encoder_config = ProductTextEncoderConfig(vocab_size,
                                                                    embed_init)
