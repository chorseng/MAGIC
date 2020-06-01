import torch
import random

from config import GlobalConfig, DatasetConfig
from constant import SOS_ID
from model import ToHidden, TextDecoder
from model import Similarity
from util import mask_nll_loss, get_mask


def text_loss(to_hidden: ToHidden, text_decoder: TextDecoder, text_length: int,
              context, target, target_length, hiddens,
              encode_knowledge_func=None, teacher_forcing_ratio=1.0):
    """Text loss.

    Args:
        to_hidden (ToHidden): Context to hidden.
        text_decoder (TextDecoder): Text decoder.
        text_length (int): Text length.
        context: Context (batch_size, ContextEncoderConfig.output_size).
        target: Target (batch_size, dialog_text_max_len)
        target_length: Target length (batch_size, ).
        encode_knowledge_func (optional): Knowledge encoding function.
        hiddens: (seq_len, batch, num_directions * hidden_size).

    Returns:
        loss: Loss.
        n_totals: Number of words which produces loss.

    """
    batch_size = context.size(0)
    loss = 0
    n_totals = 0
    mask = get_mask(text_length, target_length)
    mask = mask.transpose(0, 1)
    # (text_length, batch_size)
    hidden = to_hidden(context).to(GlobalConfig.device)
    word = SOS_ID * torch.ones(batch_size, dtype=torch.long)
    word = word.to(GlobalConfig.device)
    target = target.transpose(0, 1)
    use_teacher_forcing = random.random() < teacher_forcing_ratio

    # (text_length, batch_size)
    for i in range(text_length):
        output, hidden = text_decoder(word, hidden, hiddens,
                                      encode_knowledge_func)
        mask_loss, n_total = mask_nll_loss(output, target[i], mask[i])
        if use_teacher_forcing:
            topv, topi = output.topk(1)
            word = topi.squeeze(1).detach()
        else:
            word = target[i]
        loss += mask_loss
        n_totals += n_total
    return loss, n_totals


def recommend_loss(similarity: Similarity, batch_size: int, context,
                   pos_products, neg_products):
    """Recommend Loss.

    Args:
        similarity (Similarity):
        batch_size (int):
        context: Context.
        pos_products: Positive products. (num_pos_products, pos_images,
                      pos_product_texts, pos_product_text_lengths)
        neg_products: Negative products. (num_neg_products, neg_images,
                      neg_product_texts, neg_product_text_lengths)

    """
    ones = torch.ones(batch_size).to(GlobalConfig.device)
    zeros = torch.zeros(batch_size).to(GlobalConfig.device)

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

    # mask
    mask = get_mask(DatasetConfig.neg_images_max_num, num_neg_products)
    mask = mask.transpose(0, 1)
    # (neg_images_max_num, batch_size)

    losses = []
    for i in range(DatasetConfig.neg_images_max_num):
        neg_cos_sim = similarity(context, neg_product_texts[i],
                                 neg_product_text_lengths[i],
                                 neg_images[i])
        loss = torch.max(zeros, ones - pos_cos_sim + neg_cos_sim)
        losses.append(loss)
    losses = torch.stack(losses)
    # (neg_images_max_num, batch_size)
    loss = losses.masked_select(mask.byte()).mean()
    return loss
