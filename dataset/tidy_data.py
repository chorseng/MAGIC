"""Tidy data module."""
import copy
from os.path import isfile, join
from typing import List, Dict

from config import DatasetConfig
from constant import INTENTION_TASK, TEXT_TASK, RECOMMEND_TASK, KNOWLEDGE_TASK
from constant import KNOWLEDGE_ATTRIBUTE_SUBTASK
from constant import KNOWLEDGE_CELEBRITY_SUBTASK
from constant import KNOWLEDGE_STYLETIP_SUBTASK
from constant import TRAIN_MODE, VALID_MODE, TEST_MODE
from constant import USER_SPEAKER, SYS_SPEAKER
from dataset import RawData
from dataset.model import TidyUtterance, Utterance
from util import save_pkl, get_product_path

Dialog = List[Utterance]
TidyDialog = List[TidyUtterance]


def generate_tidy_data_file(raw_data: RawData, task: int, mode: int):
    """Generate tidy data file.

    Args:
        raw_data (RawData): Raw data.
        task (int): A single task.
        mode (int): A single mode.

    """

    # If item file already exists, then return and print a warning
    item_file_name: str = DatasetConfig.get_dialog_filename(task, mode)
    if isfile(item_file_name):
        print('Warning: Tidy data file {} exists.'.format(item_file_name))
        return

    # Get raw data dialogs according to its mode.
    dialogs: List[Dialog] = None
    if mode == TRAIN_MODE:
        dialogs = raw_data.train_dialogs
    if mode == VALID_MODE:
        dialogs = raw_data.valid_dialogs
    if mode == TEST_MODE:
        dialogs = raw_data.test_dialogs
    assert dialogs is not None

    if task & KNOWLEDGE_TASK:
        ordinal_number = {raw_data.dialog_vocab[key]: value for key, value in
                          DatasetConfig.ordinal_number.items()}

    tidy_dialogs: List[TidyDialog] = []
    for item_idx, dialog in enumerate(dialogs):
        print('Getting items from dialogs {}/{}'.format(
            item_idx + 1, len(dialogs)))
        #print(dialog)
        # Get items according to different TASKS.
        if task == INTENTION_TASK:
            # Standardize dialog first.
            std_dialog: Dialog = standardized_dialog(dialog)
            tidy_dialogs.extend(get_intention_task_items(std_dialog))
        elif task == TEXT_TASK:
            tidy_dialogs.extend(get_text_task_items(dialog))
        elif task == RECOMMEND_TASK:
            tidy_dialogs.extend(get_recommend_task_items(raw_data.image_paths,
                                                         dialog))
        elif task == KNOWLEDGE_STYLETIP_SUBTASK:
            items = get_knowledge_items(dialog, ordinal_number,
                                        KNOWLEDGE_STYLETIP_SUBTASK)
            tidy_dialogs.extend(items)
        elif task == KNOWLEDGE_ATTRIBUTE_SUBTASK:
            items = get_knowledge_items(dialog, ordinal_number,
                                        KNOWLEDGE_ATTRIBUTE_SUBTASK)
            tidy_dialogs.extend(items)
        elif task == KNOWLEDGE_CELEBRITY_SUBTASK:
            items = get_knowledge_items(dialog, ordinal_number,
                                        KNOWLEDGE_CELEBRITY_SUBTASK)
            tidy_dialogs.extend(items)

    # Save as pickle file.
    #print(tidy_dialogs[-1])
    
    save_pkl(tidy_dialogs, 'tidy_dialogs', item_file_name)


def standardized_dialog(dialog: Dialog) -> Dialog:
    """Standardized raw dialog.

    Args:
        dialog (Dialog): Raw dialog.

    Returns:
        Dialog: Standard dialog.
    
    Only retains last utterance of each speaker if more than 1 utterance consecutively
    
    """
    std_dialog: Dialog = []
    for utter in dialog:
        if not std_dialog and utter.speaker != USER_SPEAKER:
            std_dialog.append(Utterance(USER_SPEAKER, -1, [], [], []))
        if not std_dialog or utter.speaker != std_dialog[-1].speaker:
            std_dialog.append(utter)
        else:
            std_dialog[-1].utter_type = utter.utter_type
            std_dialog[-1].text += utter.text
            std_dialog[-1].pos_images += utter.pos_images
            std_dialog[-1].neg_images += utter.neg_images
    return std_dialog


def get_init_pad_utters() -> List[TidyUtterance]:
    """Get initial padding utterances.

    Returns:
        List[TidyUtterance]
    
    Creates 3 utters that alternative between user-system-user
    
    """
    utters: List[TidyUtterance] = []
    for i in range(DatasetConfig.dialog_context_size):
        if (DatasetConfig.dialog_context_size - i - 1) % 2 == 0:
            speaker = SYS_SPEAKER
        else:
            speaker = USER_SPEAKER
        utter = Utterance(speaker, -1, [], [], [])
        utters.append(TidyUtterance(utter))
    return utters


def get_intention_task_items(dialog: Dialog) -> List[TidyDialog]:
    """Get items for intention task from a single dialog.

    Args:
        dialog (Dialog): Dialog.

    Returns:
        List[TidyDialog]: Extracted tidy dialogs.
    
    Appends utters to utterances, then appends every 3 system-user-system set of utterancses to dialogs 
    
    """

    dialogs: List[TidyDialog] = []
    utterances = get_init_pad_utters()

    for utter in dialog:
        if utter.speaker == USER_SPEAKER:
            utterances.append(TidyUtterance(utter))
        if utter.speaker == SYS_SPEAKER:
            utterances.append(TidyUtterance(utter))
            utterances = utterances[-(DatasetConfig.dialog_context_size + 1):]
            dialogs.append(copy.copy(utterances))
    return dialogs


def get_text_task_items(dialog: Dialog) -> List[TidyDialog]:
    """Get items for text task from a single dialog.

    Args:
        dialog (Dialog): Dialog.

    Returns:
        List[TidyDialog]: Extracted tidy dialogs.

    """

    dialogs: List[TidyDialog] = []
    utterances = get_init_pad_utters()
    utter_type = None
    sys_responses: List[Utterance] = []
    context_size = DatasetConfig.dialog_context_size

    for utter in dialog:
        #print(utter.speaker, len(sys_responses))
        if utter.speaker == USER_SPEAKER:
            # The first utterance of three consecutive system responses must be
            # a simple response, and after getting this simple response dialog.
            # The other two responses should be in the candidate context.
            if len(sys_responses) == 3:
                for idx, response in enumerate(sys_responses):
                    utterances.append(TidyUtterance(response))
                    if idx == 0:
                        utterances = utterances[-(context_size + 1):]
                        dialogs.append(copy.copy(utterances))
            elif sys_responses:
                # If there are no three consecutive system responses, then just
                # append them to the candidate context.
                for response in sys_responses:
                    utterances.append(TidyUtterance(response))
            sys_responses = []
            utterances.append(TidyUtterance(utter))
            utter_type = utter.utter_type
        elif utter.speaker == SYS_SPEAKER:
            # If the type of last user utterance is in utterance_text_types
            # then it's also a simple response
            if utter_type in DatasetConfig.utterance_text_types: #or \
                    #utter_type in DatasetConfig.utterance_text_recommend_types:
                utterances.append(TidyUtterance(utter))
                utterances = utterances[-(context_size + 1):]
                #print('Appending utterances: ', utterances)
                dialogs.append(copy.copy(utterances))
                #print('# dialogs: ', len(dialogs)) 
                utter_type = None
            else:
                sys_responses.append(utter)
        

    if len(sys_responses) == 3:
        utterances.append(TidyUtterance(sys_responses[0]))
        utterances = utterances[-(context_size + 1):]
        dialogs.append(copy.copy(utterances))
    return dialogs


def get_valid_image(image_paths: List[str],
                    images: List[int]) -> List[int]:
    """ Get valid images in `images`.

    Args:
        image_paths (List[str]): Image paths.
        images (List[int]): Images.

    Returns:
        valid_images (List[int]): Valid images.

    """
    res_images = []
    for image_id in images:
        image_path = join(DatasetConfig.image_data_directory,
                          image_paths[image_id])
        if not isfile(image_path):
            continue
        product_path = get_product_path(image_paths[image_id])
        if not isfile(product_path):
            continue
        res_images.append(image_id)
    return res_images


def get_recommend_task_items(
        image_paths: List[str], dialog: Dialog) -> List[TidyDialog]:
    """Get items for recommend task from a single dialog.

    Args:
        image_paths (List[str]): Image paths.
        dialog (Dialog): Dialog.

    Returns:
        List[TidyDialog]: Extracted tidy dialogs.
    
    
    
    """

    dialogs: List[TidyDialog] = []
    utterances = get_init_pad_utters()
    utter_type = None
    sys_responses: List[Utterance] = []
    context_size = DatasetConfig.dialog_context_size

    for utter in dialog:
        if utter.speaker == USER_SPEAKER:
            selected_idx = -1
            for idx, response in enumerate(sys_responses):
                pos_images = get_valid_image(image_paths, response.pos_images)
                if pos_images:
                    neg_images = get_valid_image(image_paths,
                                                 response.neg_images)
                    if neg_images:
                        response.pos_images = pos_images
                        response.neg_images = neg_images
                        utterances.append(TidyUtterance(response))
                        utterances = utterances[-(context_size + 1):]
                        dialogs.append(copy.copy(utterances))
                        selected_idx = idx
                        break
            for response in sys_responses[selected_idx + 1:]:
                utterances.append(TidyUtterance(response))
            sys_responses = []
            utterances.append(TidyUtterance(utter))
            utter_type = utter.utter_type
        elif utter.speaker == SYS_SPEAKER:
            if utter_type in DatasetConfig.utterance_recommend_types:
                sys_responses.append(utter)
            else:
                utterances.append(TidyUtterance(utter))

    for response in sys_responses:
        pos_images = get_valid_image(image_paths, response.pos_images)
        if pos_images:
            neg_images = get_valid_image(image_paths, response.neg_images)
            if neg_images:
                response.pos_images = pos_images
                response.neg_images = neg_images
                utterances.append(TidyUtterance(response))
                utterances = utterances[-(context_size + 1):]
                dialogs.append(copy.copy(utterances))
                break

    return dialogs


def get_products(order_words, text, products):
    result = []
    if products:
        for word in text:
            if word in order_words:
                order = order_words[word]
                if order < len(products):
                    product = products[order]
                    result.append(product)
    if not result:
        if len(products) > 0:
            result.append(products[0])
        else:
            result.append(0)
    return result


def get_knowledge_items(dialog: Dialog, ordinal_number: Dict[int, int],
                        task: int) -> List[TidyDialog]:
    """Get items for knowledge task from a single dialog.

    Args:
        dialog (Dialog): Dialog.
        ordinal_number (Dict[int, int]): Ordinal numbers.
        task (int): Task.

    Returns:
        List[TidyDialog]: Extracted tidy dialogs.

    """
    expected_utter_types = {}
    if task == KNOWLEDGE_STYLETIP_SUBTASK:
        expected_utter_types = DatasetConfig.utterance_knowledge_styletip_types
    elif task == KNOWLEDGE_ATTRIBUTE_SUBTASK:
        expected_utter_types = DatasetConfig.utterance_knowledge_attribute_types
    elif task == KNOWLEDGE_CELEBRITY_SUBTASK:
        expected_utter_types = DatasetConfig.utterance_knowledge_celebrity_types

    dialogs: List[TidyDialog] = []
    utterances = get_init_pad_utters()
    context_size = DatasetConfig.dialog_context_size
    utter_type = None
    has_shown = False
    products = []
    selected_products = []
    for utter in dialog:
        pos_images = [image for image in utter.pos_images if image > 0]

        if utter.speaker == USER_SPEAKER:
            utterances.append(TidyUtterance(utter))
            selected_products = get_products(ordinal_number, utter.text,
                                             products)
            utter_type = utter.utter_type
        elif utter.speaker == SYS_SPEAKER:
            desc = task == KNOWLEDGE_ATTRIBUTE_SUBTASK and has_shown and \
                   utter_type in DatasetConfig.utterance_recommend_types and \
                   len(utter.text) > 10
            if utter_type in expected_utter_types or desc:
                if desc:
                    selected_products = get_products(ordinal_number, utter.text,
                                                     products)
                utterances = utterances[-context_size:]
                text = copy.deepcopy(utter.text)
                special_utter = Utterance(utter.speaker,
                                          utter.utter_type,
                                          text,
                                          selected_products,
                                          [])
                special_utter = TidyUtterance(special_utter)
                dialogs.append(copy.deepcopy(utterances + [special_utter]))
                utter_type = None
            utterances.append(TidyUtterance(utter))
            has_shown = False
            if pos_images:
                products = pos_images
                if utter_type in DatasetConfig.utterance_recommend_types:
                    has_shown = True
    return dialogs
