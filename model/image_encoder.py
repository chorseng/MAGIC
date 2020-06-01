import torch
from torch import nn
from torchvision.models import resnet18

from config import GlobalConfig
from config.model_config import ImageEncoderConfig


class ImageEncoder(nn.Module):
    """Image Encoder."""

    def __init__(self, config: ImageEncoderConfig):
        super(ImageEncoder, self).__init__()
        self.cnn = resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.cnn = self.cnn.to(GlobalConfig.device)
        self.image_size = config.image_size
        self.num_channels = config.num_channels
        image_feat_size = (self.image_size ** 2) * self.num_channels
        self.linear = nn.Linear(config.text_feat_size, image_feat_size)
        self.linear = self.linear.to(GlobalConfig.device)
        # dim = 2
        self.softmax = nn.Softmax(dim=2).to(GlobalConfig.device)

    def forward(self, image, encoded_text):
        """

        Args:
            image: (batch_size, 3, image_size, image_size)
            encoded_text: (batch_size,
                           num_layers * num_directions * hidden_size)

        Returns:
            encoded_image: (batch_size, image_feat_size)

        """
        batch_size = image.size(0)

        image = self.cnn(image)
        # (batch_size, self.num_channels, self.image_size, self.image_size)

        score = self.linear(encoded_text)
        # (batch_size, self.num_channels * self.image_size * self.image_size)
        score = score.view(batch_size, self.num_channels,
                           self.image_size * self.image_size)
        # (batch_size, self.num_channels, self.image_size * self.image_size)
        score = self.softmax(score)
        # SOFTMAX IS PERFORMED TO DIM 2

        image = image.view(
            (batch_size, self.num_channels, self.image_size * self.image_size))
        # (batch_size, self.num_channels, self.image_size * self.image_size)
        image = torch.mul(image, score)
        # (batch_size, self.num_channels, self.image_size * self.image_size)
        image_feat = image.sum(2)
        # (batch_size, num_channels)
        return image_feat


if __name__ == '__main__':
    cnn = resnet18(pretrained=True)
