"""Dataset configurations."""
from os.path import join

from torchvision import transforms

from constant import TASK_STR, MODE_STR


class DatasetConfig():
    """Dataset configurations."""

    data_directory = '/home/chorseng/fashion_data'
    #glove_file = join(data_directory, 'glove.txt')
    glove_file = '/home/chorseng/data/glove.txt')
    url2img = join(data_directory, 'url2img.txt')

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

    utterance_type_size = 18
    utterance_type_dict = {
        'unknown': 0,
        'ask_attribute': 1,
        'buy': 2,
        'celebrity': 3,
        'do_not_like_earlier_show_result': 4,
        'do_not_like_n_show_result': 4,
        'do_not_like_show_result': 4,
        'exit-message': 5,
        'filter_results': 6,
        'give-criteria': 7,
        'give-part-criteria': 8,
        'go_with': 9,
        'greeting': 10,
        'like_earlier_show_result': 11,
        'like_n_show_result': 11,
        'like_show_result': 11,
        'show_orientation': 12,
        'show_similar_to': 13,
        'sort_results': 14,
        'suited_for': 15,
        'switch-synset': 16,
        'user-info': 17
    }
    utterance_text_types = {2, 5, 8, 10}
    utterance_text_recommend_types = {7, 12, 13}
    utterance_recommend_types = {4, 7, 11, 12, 13}
    utterance_knowledge_styletip_types = {9}
    utterance_knowledge_celebrity_types = {3}
    utterance_knowledge_attribute_types = {1, 15}

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
