from typing import List

import torch

from config import GlobalConfig, DatasetConfig
from constant import SOS_ID, EOS_ID
from model import Similarity, ToHidden, TextDecoder
from util import get_mask


def recommend_eval(similarity: Similarity, batch_size: int, context,
                   pos_products, neg_products):
    """Recommend Evaluation.

    Args:
        similarity (Similarity):
        batch_size (int):
        context: Context.
        pos_products: Positive products. (num_pos_products, pos_images,
                      pos_product_texts, pos_product_text_lengths)
        neg_products: Negative products. (num_neg_products, neg_images,
                      neg_product_texts, neg_product_text_lengths)

    """

    (num_pos_products, pos_images, pos_product_texts,
     pos_product_text_lengths) = pos_products
    (num_neg_products, neg_images, neg_product_texts,
     neg_product_text_lengths) = neg_products

    # Sizes:
    # num_pos_products: (batch_size, )
    # pos_images: (batch_size, pos_images_max_num, 3, image_size, image_size)
    # pos_product_texts: (batch_size, pos_images_max_num, product_text_max_len)
    # pos_product_text_lengths: (batch_size, pos_images_max_num)
    #
    # num_neg_products: (batch_size, )
    # neg_images: (batch_size, neg_images_max_num, 3, image_size, image_size)
    # neg_product_texts: (batch_size, neg_images_max_num, product_text_max_len)
    # neg_product_text_lengths: (batch_size, neg_images_max_num)

    # num_pos_products = num_pos_products.to(GlobalConfig.device)
    pos_images = pos_images.to(GlobalConfig.device)
    pos_product_texts = pos_product_texts.to(GlobalConfig.device)
    pos_product_text_lengths = pos_product_text_lengths.to(GlobalConfig.device)
    pos_images.transpose_(0, 1)
    pos_product_texts.transpose_(0, 1)
    pos_product_text_lengths.transpose_(0, 1)
    # pos_images: (pos_images_max_num, batch_size, 3, image_size, image_size)
    # pos_product_texts: (pos_images_max_num, batch_size, product_text_max_len)
    # pos_product_text_lengths: (pos_images_max_num, batch_size)

    num_neg_products = num_neg_products.to(GlobalConfig.device)
    neg_images = neg_images.to(GlobalConfig.device)
    neg_product_texts = neg_product_texts.to(GlobalConfig.device)
    neg_product_text_lengths = neg_product_text_lengths.to(GlobalConfig.device)
    neg_images.transpose_(0, 1)
    neg_product_texts.transpose_(0, 1)
    neg_product_text_lengths.transpose_(0, 1)
    # neg_images: (neg_images_max_num, batch_size, 3, image_size, image_size)
    # neg_product_texts: (neg_images_max_num, batch_size, product_text_max_len)
    # neg_product_text_lengths: (neg_images_max_num, batch_size)

    pos_cos_sim = similarity(context, pos_product_texts[0],
                             pos_product_text_lengths[0],
                             pos_images[0])
    # Mask.
    mask = get_mask(DatasetConfig.neg_images_max_num, num_neg_products)
    mask = mask.transpose(0, 1).long()
    # (neg_images_max_num, batch_size)

    rank = torch.zeros(batch_size, dtype=torch.long).to(GlobalConfig.device)
    for i in range(DatasetConfig.neg_images_max_num):
        neg_cos_sim = similarity(context, neg_product_texts[i],
                                 neg_product_text_lengths[i],
                                 neg_images[i])
        rank += torch.lt(pos_cos_sim, neg_cos_sim).long() * mask[i]

    num_rank = [0] * (DatasetConfig.neg_images_max_num + 1)
    for i in range(batch_size):
        num_rank[rank[i]] += 1

    return torch.tensor(num_rank).to(GlobalConfig.device)


def text_eval(to_hidden: ToHidden, text_decoder: TextDecoder, text_length: int,
              id2word: List[str], context, target, hiddens,
              encode_knowledge_func=None, output_file=None):
    """Text loss.

    Args:
        to_hidden (ToHidden): Context to hidden.
        text_decoder (TextDecoder): Text decoder.
        text_length (int): Text length.
        id2word (List[str]): Word id to str.
        context: Context (batch_size, ContextEncoderConfig.output_size).
        target: Target (batch_size, dialog_text_max_len)
        encode_knowledge_func (optional): Knowledge encoding function.
        output_file (optional): Output file.

    """
    batch_size = context.size(0)
    # (text_length, batch_size)
    hidden = to_hidden(context)
    word = SOS_ID * torch.ones(batch_size, dtype=torch.long)
    word = word.to(GlobalConfig.device)
    target = target.transpose(0, 1)
    # (text_length, batch_size)

    all_tokens = torch.zeros((text_length, batch_size), dtype=torch.long)
    all_tokens = all_tokens.to(GlobalConfig.device)
    all_scores = torch.zeros((text_length, batch_size))
    all_scores = all_scores.to(GlobalConfig.device)

    for i in range(text_length):
        output, hidden = text_decoder(word, hidden,
                                      hiddens, encode_knowledge_func)
        score, word = torch.max(output, dim=1)
        all_tokens[i], all_scores[i] = word, score

    for j in range(batch_size):
        str_pred = []
        str_true = []
        for i in range(text_length):
            if all_tokens[i][j] == EOS_ID:
                break
            word = id2word[all_tokens[i][j]]
            str_pred.append(word)

        for i in range(text_length):
            if target[i][j] == EOS_ID:
                break
            word = id2word[target[i][j]]
            str_true.append(word)
        line = "{}\t{}".format(' '.join(str_pred), ' '.join(str_true))
        if output_file:
            output_file.write(line + '\n')
        else:
            print(line)

    return all_tokens
