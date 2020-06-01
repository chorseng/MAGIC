from operator import itemgetter
from queue import PriorityQueue
from typing import List, Dict

import torch

from config import GlobalConfig, BeamSearchConfig
from constant import SOS_ID, EOS_ID
from model import ToHidden, TextDecoder


class BeamSearchNode:
    def __init__(self, hidden, prev_node, word_id, log_prob, length):
        self.hidden = hidden
        self.prev_node = prev_node
        self.word_id = word_id
        self.log_prob = log_prob
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.log_prob / float(self.length - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.log_prob < other.log_prob


class BeamSearchDecoder:
    def __init__(self, to_hidden: ToHidden, decoder: TextDecoder,
                 config: BeamSearchConfig, vocab: Dict[str, int],
                 encode_knowledge_func=None):
        # Models.
        self.to_hidden = to_hidden
        self.decoder: TextDecoder = decoder

        # Configs.
        self.beam_size: int = config.beam_size
        self.text_length: int = config.text_len
        self.top_k: int = config.top_k
        self.give_up: int = config.give_up

        # Vocabulary.
        self.id2word: List[str] = [None] * len(vocab)
        for word, wid in vocab.items():
            self.id2word[wid] = word

    def beam_decode(self, context, target, hiddens, encode_knowledge_func=None):
        """Beam decode.
        Args:
            context: Context (batch_size, ContextEncoderConfig.output_size).
            target: Target text (batch_size, dialog_text_max_len).
        Returns:
            decoded_utterances: Decoded utterances (batch_size, top_k,
                                                    dialog_text_max_len).
        """
        batch_size = context.size(0)
        decoded_utterances = []

        for batch_id in range(batch_size):
            hidden = self.to_hidden(context[batch_id].unsqueeze(0)).view(1, -1)
            # hidden: (1, hidden_size)

            word = torch.tensor([SOS_ID])
            word = word.to(GlobalConfig.device)
            # word: (1,)

            end_nodes = []
            node = BeamSearchNode(hidden, None, word, 0, 1)
            queue = PriorityQueue()
            queue.put((-node.eval(), node))

            # Start searching.
            while True:
                # Give up when decoding takes too long.
                if queue.qsize() > self.give_up:
                    break

                # Fetch the best node.
                best_score, best_node = queue.get()
                word = best_node.word_id
                hidden = best_node.hidden

                if best_node.word_id == EOS_ID:
                    if best_node.prev_node is not None:
                        end_nodes.append((best_score, best_node))
                        if len(end_nodes) >= self.top_k:
                            break
                        else:
                            continue

                output, hidden = self.decoder(word, hidden,
                                              hiddens[:, batch_id].unsqueeze(1),
                                              encode_knowledge_func, batch_id)
                # output: (1, vocab_size)
                # hidden: (1, hidden_size)

                log_probs, word_ids = torch.topk(output[0], self.beam_size)
                next_nodes = []

                for k in range(self.beam_size):
                    word_id = torch.tensor([word_ids[k].item()]).to(
                        GlobalConfig.device)
                    log_prob = log_probs[k].item()

                    node = BeamSearchNode(hidden, best_node, word_id,
                                          best_node.log_prob + log_prob,
                                          best_node.length + 1)
                    next_nodes.append((-node.eval(), node))

                for i in range(len(next_nodes)):
                    next_score, next_node = next_nodes[i]
                    queue.put((next_score, next_node))

            if len(end_nodes) == 0:
                end_nodes = [queue.get() for _ in range(self.top_k)]

            utterances = []
            predict_utters = []
            for best_score, best_node in sorted(end_nodes, key=itemgetter(0)):
                utterance = [best_node.word_id]
                while best_node.prev_node is not None:
                    best_node = best_node.prev_node
                    utterance.append(best_node.word_id)
                utterance = utterance[::-1]
                utterances.append(utterance)
                predict_utters.append(' '.join([self.id2word[wid]
                                                for wid in utterance]))

            # Print predicted utterances.
            for predict_utter in predict_utters:
                print('>', predict_utter)

            # Print ground truth.
            target_utter = []
            for i in range(self.text_length):
                if target[batch_id][i] == EOS_ID:
                    break
                word = self.id2word[target[batch_id][i]]
                target_utter.append(word)
            print('<', ' '.join(target_utter))

            decoded_utterances.append(utterances)

        return decoded_utterances
