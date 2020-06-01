from torch import nn

from config import GlobalConfig
from config.model_config import IntentionConfig


class Intention(nn.Module):
    def __init__(self, config: IntentionConfig):
        super(Intention, self).__init__()
        self.fcs = []
        for i in range(len(config.fc_sizes) - 1):
            linear = nn.Linear(config.fc_sizes[i], config.fc_sizes[i + 1])
            linear = linear.to(GlobalConfig.device)
            self.fcs.append(linear)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, context):
        """Forward.

        Args:
            context: Context (batch_size, ContextEncoderConfig.output_size).

        Returns:
            output: (batch_size, utterance_type_size)

        """

        x = context
        # (batch_size, ContextEncoderConfig.output_size)
        for fc in self.fcs:
            x = fc(x)
        # (batch_size, utterance_type_size)

        output = self.softmax(x)
        # (batch_size, utterance_type_size)
        return output
