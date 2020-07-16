"""Dataset configurations."""
from os.path import join

from torchvision import transforms

from constant import TASK_STR, MODE_STR


class DatasetConfig():
    """Dataset configurations."""

    data_directory = '/home/chorseng/fashion_data'
    #glove_file = join(data_directory, 'glove.txt')
    glove_file = '/home/chorseng/data/glove.txt'
    url2img = '/home/chorseng/data/url2img.txt'

    dialog_directory = join(data_directory, 'dialogs/')
    train_dialog_data_directory = join(dialog_directory, 'train/')
    valid_dialog_data_directory = join(dialog_directory, 'valid/')
    test_dialog_data_directory = join(dialog_directory, 'test/')

    knowledge_data_directory = join(data_directory, 'knowledge/')
    product_data_directory = join(knowledge_data_directory,
                                  'products_format/')

    styletips_data_file = join(knowledge_data_directory,
                               'styletip/styletips_synset.txt')
    celebrity_data_file = join(knowledge_data_directory,
                               'celebrity/celebrity_distribution.json')

    image_data_directory = join(data_directory, 'images/')

    dump_dir = '/home/chorseng/fashion_data/dump_dir'
    common_raw_data_file = join(dump_dir, 'common_raw_data.pkl')
    knowledge_data_file = join(dump_dir, 'knowledge_data.pkl')

    train_raw_data_file = join(dump_dir, 'train_raw_data.pkl')
    valid_raw_data_file = join(dump_dir, 'valid_raw_data.pkl')
    test_raw_data_file = join(dump_dir, 'test_raw_data.pkl')

    # Temporarily, they share the same dialog file
    intention_train_dialog_file = join(dump_dir, 'intention_train_dialog.pkl')
    intention_valid_dialog_file = join(dump_dir, 'intention_valid_dialog.pkl')
    intention_test_dialog_file = join(dump_dir, 'intention_test_dialog.pkl')

    text_train_dialog_file = join(dump_dir, 'text_train_dialog.pkl')
    text_valid_dialog_file = join(dump_dir, 'text_valid_dialog.pkl')
    text_test_dialog_file = join(dump_dir, 'text_test_dialog.pkl')

    recommend_train_dialog_file = join(dump_dir,
                                       'recommend_train_dialog_file.pkl')
    recommend_valid_dialog_file = join(dump_dir,
                                       'recommend_valid_dialog_file.pkl')
    recommend_test_dialog_file = join(dump_dir,
                                      'recommend_test_dialog_file.pkl')

    knowledge_styletip_train_dialog_file = join(
        dump_dir, 'knowledge_styletip_train_dialog_file.pkl')
    knowledge_styletip_valid_dialog_file = join(
        dump_dir, 'knowledge_styletip_valid_dialog_file.pkl')
    knowledge_styletip_test_dialog_file = join(
        dump_dir, 'knowledge_styletip_test_dialog_file.pkl')

    knowledge_attribute_train_dialog_file = join(
        dump_dir, 'knowledge_attribute_train_dialog_file.pkl')
    knowledge_attribute_valid_dialog_file = join(
        dump_dir, 'knowledge_attribute_valid_dialog_file.pkl')
    knowledge_attribute_test_dialog_file = join(
        dump_dir, 'knowledge_attribute_test_dialog_file.pkl')

    knowledge_celebrity_train_dialog_file = join(
        dump_dir, 'knowledge_celebrity_train_dialog_file.pkl')
    knowledge_celebrity_valid_dialog_file = join(
        dump_dir, 'knowledge_celebrity_valid_dialog_file.pkl')
    knowledge_celebrity_test_dialog_file = join(
        dump_dir, 'knowledge_celebrity_test_dialog_file.pkl')

    @staticmethod
    def get_dialog_filename(task: int, mode: int) -> str:
        """Get dialog file name according to its task and mode.

        Note:
            task and mode should be the power of 2.

        Args:
            task (int): Task.
            mode (int): Mode.

        Returns:
            str: Dialog file name.

        """
        task_str = TASK_STR[task]
        mode_str = MODE_STR[mode]
        # Note: Using reflection, be careful!
        result = getattr(DatasetConfig,
                         '{}_{}_dialog_file'.format(task_str, mode_str))
        return result

    dialog_text_cutoff = 4
    dialog_context_size = 2
    dialog_text_max_len = 30
    pos_images_max_num = 1
    neg_images_max_num = 4

    utterance_type_size = 46
    utterance_type_dict = {
        'ERR': 0,
        'DA': 1
        """
        'DA:ASK:ADD_TO_CART': 1,
        'DA:ASK:CHECK': 2,
        'DA:ASK:COMPARE': 3,
        'DA:ASK:COUNT': 4,
        'DA:ASK:DISPREFER': 5,
        'DA:ASK:GET': 6,
        'DA:ASK:PREFER': 7,
        'DA:ASK:REFINE': 8,
        'DA:ASK:ROTATE': 9,
        'DA:CONFIRM:ADD_TO_CART': 10,
        'DA:CONFIRM:CHECK': 11,
        'DA:CONFIRM:COMPARE': 12,
        'DA:CONFIRM:COUNT': 13,
        'DA:CONFIRM:DISPREFER': 14,
        'DA:CONFIRM:GET': 15,
        'DA:CONFIRM:PREFER': 16,
        'DA:CONFIRM:REFINE': 17,
        'DA:CONFIRM:ROTATE': 18,
        'DA:INFORM:ADD_TO_CART': 19,
        'DA:INFORM:CHECK': 20,
        'DA:INFORM:COMPARE': 21,
        'DA:INFORM:COUNT': 22,
        'DA:INFORM:DISPREFER': 23,
        'DA:INFORM:GET': 24,
        'DA:INFORM:PREFER': 25,
        'DA:INFORM:REFINE': 26,
        'DA:INFORM:ROTATE': 27,
        'DA:PROMPT:ADD_TO_CART': 28,
        'DA:PROMPT:CHECK': 29,
        'DA:PROMPT:COMPARE': 30,
        'DA:PROMPT:COUNT': 31,
        'DA:PROMPT:DISPREFER': 32,
        'DA:PROMPT:GET': 33,
        'DA:PROMPT:PREFER': 34,
        'DA:PROMPT:REFINE': 35,
        'DA:REQUEST:ROTATE': 36,
        'DA:REQUEST:ADD_TO_CART': 37,
        'DA:REQUEST:CHECK': 38,
        'DA:REQUEST:COMPARE': 39,
        'DA:REQUEST:COUNT': 40,
        'DA:REQUEST:DISPREFER': 41,
        'DA:REQUEST:GET': 42,
        'DA:REQUEST:PREFER': 43,
        'DA:REQUEST:REFINE': 44,
        'DA:REQUEST:ROTATE': 45,
        """
    }
    utterance_text_types = {0}
    #utterance_text_recommend_types = {7, 12, 13}
    #utterance_recommend_types = {4, 7, 11, 12, 13}
    #utterance_knowledge_styletip_types = {9}
    #utterance_knowledge_celebrity_types = {3}
    utterance_knowledge_attribute_types = {1}
    #utterance_knowledge_attribute_types = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                          16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                                          29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                                          42, 43, 44, 45}

    image_size = 64

    # image transform
    transform = transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    product_text_max_len = 30

    product_attributes = [
        "available_sizes", "brand", "care", "color", "color_name", "fit",
        "gender", "length", "material", "name", "neck", "price", "review",
        "reviewstars", "size_fit", "sleeves", "style", "taxonomy", "type"
    ]
    num_keys = len(product_attributes)
    product_value_cutoff = 10

    ordinal_number = {
        '1st': 0,
        '2nd': 1,
        '3rd': 2,
        '4th': 3,
        '5th': 4
    }
