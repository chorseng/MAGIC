import torch
import torch.nn as nn

from config import GlobalConfig
from config.model_config import TextDecoderConfig, KnowledgeTextDecoderConfig


class TextDecoder(nn.Module):
    def __init__(self, config: TextDecoderConfig):
        super(TextDecoder, self).__init__()

        # Embedding.
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.word_embed_size,
            padding_idx=config.pad_index)
        self.embedding = self.embedding.to(GlobalConfig.device)
        # Initial embedding.
        if config.embed_init is not None:
            self.embedding = self.embedding.from_pretrained(
                config.embed_init, freeze=False)
        if isinstance(config, KnowledgeTextDecoderConfig):
            self.fc = nn.Linear(config.fc_in_size, config.embed_size)
        # GRU.
        self.gru = nn.GRU(config.embed_size, config.hidden_size,
                          num_layers=config.num_layers)
        self.gru = self.gru.to(GlobalConfig.device)
        # Linear.
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)
        self.linear = self.linear.to(GlobalConfig.device)
        # Softmax.
        self.softmax = nn.Softmax(dim=1)
        self.softmax = self.softmax.to(GlobalConfig.device)

    def forward(self, word, hidden, enc_hiddens, encode_knowledge_func=None,
                batch_id=None):
        """Forward.

        Args:
            word: Word (batch_size).
            hidden: Hidden state (batch_size, hidden_size).
            enc_hiddens: (seq_len, batch_size, hidden_size)
            encode_knowledge_func (optional): Knowledge encoding function.

        Returns:
            output: Output (batch_size, vocab_size).
            hidden: Next hidden state (batch_size, hidden_size).

        """
        embed = self.embedding(word)

        if encode_knowledge_func is not None:
            knowledge = encode_knowledge_func(hidden, batch_id)
            # enc_hiddens: (seq_len, batch_size, hidden_size)
            # hidden: (batch_size, hidden_size)
            attention = torch.matmul(enc_hiddens.unsqueeze(2),
                                     hidden.unsqueeze(0).unsqueeze(3))
            attention = attention.squeeze(2).squeeze(2)
            attention = attention.transpose(0, 1)
            attention = self.softmax(attention)
            enc_hidden = torch.matmul(attention.unsqueeze(1),
                                      enc_hiddens.transpose(0, 1)).squeeze(1)
            embed = self.fc(torch.cat((embed, knowledge, enc_hidden), dim=1))

        output, hidden = self.gru(embed.unsqueeze(0), hidden.unsqueeze(0))
        output = self.linear(output[0])
        output = self.softmax(output)
        hidden = hidden[0]
        return output, hidden


class ToHidden(nn.Module):
    def __init__(self, config: TextDecoderConfig):
        super(ToHidden, self).__init__()
        self.fcs = []
        for i in range(len(config.to_hidden_fc_sizes) - 1):
            linear = nn.Linear(config.to_hidden_fc_sizes[i],
                               config.to_hidden_fc_sizes[i + 1])
            linear = linear.to(GlobalConfig.device)
            self.fcs.append(linear)

    def forward(self, x):
        for fc in self.fcs:
            x = fc(x)
        return x
