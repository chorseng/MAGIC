import torch
from torch import nn
from torch.nn.functional import cosine_similarity

from config import GlobalConfig
from config.model_config import SimilarityConfig
from constant import SOS_ID
from model import TextEncoder, ImageEncoder


class Similarity(nn.Module):
    def __init__(self, config: SimilarityConfig):
        super(Similarity, self).__init__()

        self.text_encoder = TextEncoder(config.product_text_encoder_config)
        self.text_encoder = self.text_encoder.to(GlobalConfig.device)

        self.image_encoder = ImageEncoder(config.product_image_encoder_config)
        self.image_encoder = self.image_encoder.to(GlobalConfig.device)

        self.linear = nn.Linear(config.mm_size, config.context_vector_size)
        self.linear = self.linear.to(GlobalConfig.device)

    def forward(self, context, text, text_length, image):
        """Forward.

        Args:
            context: Context (batch_size, ContextEncoderConfig.output_size).
            text: Product text (batch_size, product_text_max_len).
            text_length: Product text length (batch_size, ).
            image: Product image (batch_size, 3, image_size, image_size).

        Returns:

        """

        batch_size = context.size(0)
        sos = SOS_ID * torch.ones(batch_size, dtype=torch.long).view(-1, 1).to(
            GlobalConfig.device)
        # (batch_size)

        # Concat SOS.
        text = torch.cat((sos, text), 1).to(GlobalConfig.device)
        # (batch_size, product_text_max_len)
        text_length += 1
        # (batch_size, )

        encoded_text, _ = self.text_encoder(text, text_length)
        # (batch_size, text_feat_size)
        encoded_image = self.image_encoder(image, encoded_text)
        # (batch_size, image_feat_size)

        mm = torch.cat((encoded_text, encoded_image), 1)
        mm = mm.to(GlobalConfig.device)
        mm = self.linear(mm)
        return cosine_similarity(context, mm)
