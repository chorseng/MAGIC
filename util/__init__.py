"""Utilities."""

import pickle as pkl
from os.path import splitext, join
from typing import *
from copy import deepcopy

import torch

from config import DatasetConfig, GlobalConfig
from constant import PAD_ID, EOS_ID


def save_pkl(obj, obj_name, pkl_file):
    """Save object to a .pkl file.

    Args:
        obj (object): Object.
        obj_name (str): Object name.
        pkl_file (str): Pickle file name.

    Returns:
        None

    """
    print('Saving {} to {}...'.format(obj_name, pkl_file))
    with open(pkl_file, 'wb') as file:
        pkl.dump(obj, file)
    print('Saved.')


def load_pkl(pkl_file):
    """Load object from a .pkl file.

    Args:
        pkl_file (str): Pickle file name.

    Returns:
        obj (object): Object.

    """
    print('Loading {} ...'.format(pkl_file))
    with open(pkl_file, 'rb') as file:
        obj = pkl.load(file)
    print('Loaded.')
    return obj


def pad_or_clip_text(text: List[int],
                     max_len: int) -> Tuple[List[int], int]:
    """Pad or clip text.

    Args:
        text (List[int]): Word id list.
        max_len (int): Maximum length.

    Returns:
        Tuple[List[int], int]: Padded word list of length max_len and its
        actual length.

    """
    text = deepcopy(text)
    if len(text) > max_len - 1:
        text = text[:max_len - 1]
    text.append(EOS_ID)
    text_len = len(text)
    text += [PAD_ID] * (max_len - len(text))
    return text, text_len


def pad_or_clip_images(images: List[int],
                       max_len: int) -> Tuple[List[int], int]:
    """Pad or clip images.

    Args:
        images (List[int]): Image id list.
        max_len (int): Maximum length.

    Returns:
        Tuple[List[int], int]: Padded image list of length max_len and its
        actual length.

    """
    images = deepcopy(images)
    if len(images) > max_len:
        images = images[:max_len]
    num_images = len(images)
    images += [0] * (max_len - len(images))
    return images, num_images


def get_embed_init(glove: List[List[float]], vocab_size: int):
    """Get initial embedding.

    Args:
        glove (List[List[float]]): GloVe.
        vocab_size (int): Vocabulary size.

    Returns:
        Initial embedding (vocab_size, embed_size).

    """
    embed = [None] * vocab_size
    for idx in range(vocab_size):
        vec = glove[idx]
        if vec is None:
            vec = torch.zeros(300)
            if idx != PAD_ID:
                vec.uniform_(-0.25, 0.25)
        else:
            vec = torch.tensor(vec)
        embed[idx] = vec
    embed = torch.stack(embed)
    return embed


def get_product_path(image_name: str):
    """Get product path from a given image name.

    Args:
        image_name (str): Image name.

    Returns:
        str: Corresponding product file name.

    """
    product_path = join(DatasetConfig.product_data_directory,
                        splitext(image_name)[0] + '.json')
    return product_path


def nll_loss(inp, target):
    cross_entropy = -torch.log(
        torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = cross_entropy.mean()
    loss = loss.to(GlobalConfig.device)
    return loss


def mask_nll_loss(inp, target, mask):
    # print('inp = {}'.format(inp))
    # print('target = {}'.format(target))
    # print('mask = {}'.format(mask))
    n_total = mask.sum()
    # print('n_total = {}'.format(n_total))
    if n_total.item() == 0:
        return 0, 0
    cross_entropy = -torch.log(
        torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = cross_entropy.masked_select(mask.byte()).mean()
    loss = loss.to(GlobalConfig.device)
    return loss, n_total.item()


def get_mask(length: int, target_length):
    """Get mask.

    Args:
        length (int): Length.
        target_length: Target length (batch_size, ).

    Returns:
        mask: Mask (batch_size, length).

    """
    return torch.arange(length, device=GlobalConfig.device).expand(
        target_length.size(0), length) < target_length.unsqueeze(1)


def masked_softmax(vector: torch.Tensor,
                   mask: torch.BoolTensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.BoolTensor:
    """

    Source: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

    ``torch.nn.functional.softmax(vector)`` does not work if some elements
    of ``vector`` should be masked.  This performs a softmax on just the
    non-masked portions of ``vector``.  Passing ``None`` in for the mask is
    also acceptable; you'll just get a regular softmax. ``vector`` can have
    an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions
    than ``vector``, we will unsqueeze on dimension 1 until they match.  If
    you need a different unsqueezing of your mask, do it yourself before
    passing the mask into this function. If ``memory_efficient`` is set to
    true, we will simply use a very large negative number for those masked
    positions so that the probabilities of those positions would be
    approximately 0. This is not accurate in math, but works for most cases
    and consumes less memory. In the case that the input vector is completely
    masked and ``memory_efficient`` is false, this function returns an array
    of ``0.0``. This behavior may cause ``NaN`` if this is used as the last
    layer of a model that uses categorical cross-entropy loss. Instead,
    if ``memory_efficient`` is true, this function will treat every element
    as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the
            # mask, we zero these out.
            result = torch.nn.functional.softmax(torch.mul(vector, mask),
                                                 dim=dim)
            result = torch.mul(result, mask)
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(),
                                               mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def to_str(o):
    if isinstance(o, dict):
        res = []
        for key, val in o.items():
            res.append(key)
            res.append(to_str(val))
        return ' '.join(res)
    elif isinstance(o, list):
        return ' '.join([to_str(x) for x in o])
    elif isinstance(o, str):
        return o
    return str(o)
