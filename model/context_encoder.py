import torch
from torch import nn
from config.model_config import ContextEncoderConfig
from config import GlobalConfig


class ContextEncoder(nn.Module):
    def __init__(self, config: ContextEncoderConfig):
        super(ContextEncoder, self).__init__()
        self.rnn = nn.LSTM(input_size=config.embed_size,
                           hidden_size=config.hidden_size,
                           num_layers=config.num_layers,
                           bidirectional=config.num_directions == 2,
                           batch_first=False).to(GlobalConfig.device)
        self.rnn = self.rnn.to(GlobalConfig.device)

        self.linear = nn.Linear(config.input_size, config.embed_size)
        self.linear = self.linear.to(GlobalConfig.device)

    def forward(self, context):
        """

        Args:
            context: (dialog_context_size, batch_size,
                      text_feat_size + image_feat_size)

        Returns:
            context vector: (batch_size,
                             hidden_size * num_layers * num_directions)

        """

        batch_size = context.size(1)

        context = torch.stack([self.linear(x) for x in context])
        context = context.to(GlobalConfig.device)
        # (dialog_context_size, batch_size, text_feat_size + image_feat_size)

        _, (h_n, _) = self.rnn(context)
        # (num_layers * num_directions, batch_size, hidden_size)

        output = h_n.transpose(0, 1)
        output = output.contiguous().view(batch_size, -1)
        # (batch_size, hidden_size * num_layers * num_directions)
        return output
