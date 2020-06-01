import torch.nn as nn
from config.global_config import GlobalConfig
from config.model_config import TextEncoderConfig


class TextEncoder(nn.Module):
    """ Text encoder."""

    def __init__(self, config: TextEncoderConfig):
        super(TextEncoder, self).__init__()

        # Embedding.
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embed_size,
            padding_idx=config.pad_index)
        self.embedding = self.embedding.to(GlobalConfig.device)
        # Initial embedding.
        if config.embed_init is not None:
            self.embedding = self.embedding.from_pretrained(
                config.embed_init, freeze=False)

        # LSTM.
        self.rnn = nn.LSTM(
            input_size=config.embed_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.num_directions == 2,
            batch_first=False,
            dropout=config.dropout)
        self.rnn = self.rnn.to(GlobalConfig.device)

    def forward(self, input_seq, input_lengths):
        """Forward.

        Args:
            input_seq: (batch_size, seq_len)
            input_lengths: (batch_size, )

        Returns:
            output: (batch_size, num_layers * num_directions * hidden_size)

        """
        batch_size = input_lengths.size(0)

        input_seq = input_seq.transpose(0, 1)
        # (seq_len, batch_size)

        embedded = self.embedding(input_seq)
        # (seq_len, batch_size, embed_size)

        hiddens, (h_n, _) = self.rnn(embedded)
        # hiddens: (seq_len, batch, num_directions * hidden_size)
        # h_n: (num_layers * num_directions, batch_size, hidden_size)
        h_n = h_n.transpose(0, 1)
        # h_n: (batch_size, num_layers * num_directions, hidden_size)

        output = h_n.contiguous().view(batch_size, -1)
        # (batch_size, num_layers * num_directions * hidden_size)

        return output, hiddens
