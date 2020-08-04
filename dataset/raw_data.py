"""Raw data module."""
import copy
import json
import ast
from collections import Counter, namedtuple
from os import listdir
from os.path import isfile, isdir, join
from typing import List, Tuple, Dict, Optional

from nltk import word_tokenize

from config import DatasetConfig
from constant import DIALOG_PROC_PRINT_FREQ
from constant import SYS_SPEAKER, USER_SPEAKER
from constant import TRAIN_MODE, VALID_MODE, TEST_MODE
from constant import UNK_ID, SPECIAL_TOKENS
from dataset.model import Utterance
from util import load_pkl, save_pkl

# Custom types.
CommonData = namedtuple('CommonData',
                        ['dialog_vocab', 'glove',
                         'image_url_id', 'image_paths'])
Dialog = List[Utterance]


class RawData():
    """Raw data class.

    This class extracts raw data from some data files or directories or loads
    extracted .pkl files. Data is divided into two parts: common data and mode
    specific data. Common data is always loaded whenever the project runs. Mode
    specific data is loaded when the extracted item data file of specific
    mode doesn't exist.

    Common data:
        * dialog_vocab
        * glove
        * image_url_id
        * image_paths

    Mode specific data:
        * {train, valid, test}_dialogs

    Attributes:
        mode (int): Mode (TRAIN_MODE, VALID_MODE, TEST_MODE).
        dialog_vocab (Dict[str, int]): Dialog vocabulary, word -> index.
        glove (List[Optional[List[float]]]): GloVe, index -> float vector.
        image_url_id (Dict[str, int]): Image url to index of `image_paths`.
                                       0 for unknown url.
        image_paths (List[str]): Image paths. image_paths[0] is None.

        train_dialogs (List[Dialog]): Train dialogs, if is train mode.
        valid_dialogs (List[Dialog]): Valid dialogs, if is valid mode.
        test_dialogs (List[Dialog]): Test dialogs, if is test mode.

    """

    def __init__(self, mode: int):
        # Note: For convenience, RawData loads common data (if exists) only if
        # mode is NONE_MODE

        # Attributes.
        self.mode: int = mode

        self.dialog_vocab: Dict[str, int] = None
        self.glove: List[Optional[List[float]]] = None
        self.image_url_id: Dict[str, int] = None
        self.image_paths: List[str] = None

        # Dynamic attributes.
        if self.mode & TRAIN_MODE:
            self.train_dialogs: List[Dialog] = None
        if self.mode & VALID_MODE:
            self.valid_dialogs: List[Dialog] = None
        if self.mode & TEST_MODE:
            self.test_dialogs: List[Dialog] = None

        # Check if consistency of data files.
        RawData.check_consistency(mode)

        # Read existed extracted data files.
        self.read_extracted_data()

        # If common data doesn't exist, then we need to get it.
        if not isfile(DatasetConfig.common_raw_data_file):
            common_data = RawData._get_common_data()
            self.dialog_vocab: Dict[str, int] = common_data.dialog_vocab
            self.glove: List[Optional[List[float]]] = common_data.glove
            self.image_url_id: Dict[str, int] = common_data.image_url_id
            self.image_paths: List[str] = common_data.image_paths

            # Save common data to a .pkl file.
            save_pkl(common_data, 'common_data',
                     DatasetConfig.common_raw_data_file)

        # If mode specific data doesn't exist, then we need to get it.
        if self.mode & TRAIN_MODE:
            has_data_pkl = isfile(DatasetConfig.train_raw_data_file)

            if not has_data_pkl:
                self.train_dialogs = RawData._get_dialogs(TRAIN_MODE,
                                                          self.dialog_vocab,
                                                          self.image_url_id)
                # Save common data to a .pkl file.
                save_pkl(self.train_dialogs, 'train_dialogs',
                         DatasetConfig.train_raw_data_file)

        if self.mode & VALID_MODE:
            has_data_pkl = isfile(DatasetConfig.valid_raw_data_file)

            if not has_data_pkl:
                self.valid_dialogs = RawData._get_dialogs(VALID_MODE,
                                                          self.dialog_vocab,
                                                          self.image_url_id)
                # Save common data to a .pkl file.
                save_pkl(self.valid_dialogs, 'valid_dialogs',
                         DatasetConfig.valid_raw_data_file)

        if self.mode & TEST_MODE:
            has_data_pkl = isfile(DatasetConfig.test_raw_data_file)

            if not has_data_pkl:
                self.test_dialogs = RawData._get_dialogs(TEST_MODE,
                                                         self.dialog_vocab,
                                                         self.image_url_id)
                # Save common data to a .pkl file.
                save_pkl(self.test_dialogs, 'test_dialogs',
                         DatasetConfig.test_raw_data_file)

    @staticmethod
    def check_consistency(mode: int) -> None:
        """Check data consistency.

        Args:
            mode (int): Mode (TRAIN_MODE, VALID_MODE, TEST_MODE).
                        e.g. Mode=TRAIN_MODE | VALID_MODE

        Raises:
            ValueError: Raises ValueError if data is inconsistent.

        """

        # Check if dialog data exists.
        if mode & TRAIN_MODE:
            has_data_dir = isdir(DatasetConfig.train_dialog_data_directory)
            has_data_pkl = isfile(DatasetConfig.train_raw_data_file)

            if not has_data_dir and not has_data_pkl:
                raise ValueError("No training dataset.")

        if mode & VALID_MODE:
            has_data_dir = isdir(DatasetConfig.valid_dialog_data_directory)
            has_data_pkl = isfile(DatasetConfig.valid_raw_data_file)

            if not has_data_dir and not has_data_pkl:
                raise ValueError("No validation dataset.")

        if mode & TEST_MODE:
            has_data_dir = isdir(DatasetConfig.test_dialog_data_directory)
            has_data_pkl = isfile(DatasetConfig.test_raw_data_file)

            if not has_data_dir and not has_data_pkl:
                raise ValueError("No testing dataset.")

        # Check if there's no common data but data of specific mode exists.
        if not isfile(DatasetConfig.common_raw_data_file):
            consistent = True
            if mode & TRAIN_MODE and isfile(DatasetConfig.train_raw_data_file):
                consistent = False
            if mode & VALID_MODE and isfile(DatasetConfig.valid_raw_data_file):
                consistent = False
            if mode & TEST_MODE and isfile(DatasetConfig.test_raw_data_file):
                consistent = False
            if not consistent:
                raise ValueError("Extracted common data doesn't exist"
                                 " but extracted specific mode data exists.")

            # Train and valid data is necessary to get common data.
            has_train_dir = isdir(DatasetConfig.train_dialog_data_directory)
            has_valid_dir = isdir(DatasetConfig.valid_dialog_data_directory)
            if not has_train_dir or not has_valid_dir:
                raise ValueError(
                    "Expected train and valid dialog data to extract vocab.")

    def read_extracted_data(self) -> None:
        """ Read existed data.

        Data consists of common data and specific mode data.

        """

        # Common data
        if isfile(DatasetConfig.common_raw_data_file):
            common_data: CommonData = load_pkl(
                DatasetConfig.common_raw_data_file)
            self.dialog_vocab = common_data.dialog_vocab
            self.glove = common_data.glove
            self.image_url_id = common_data.image_url_id
            self.image_paths = common_data.image_paths

        # Specific mode data
        if self.mode & TRAIN_MODE and isfile(DatasetConfig.train_raw_data_file):
            train_data = load_pkl(DatasetConfig.train_raw_data_file)
            self.train_dialogs = train_data
        if self.mode & VALID_MODE and isfile(DatasetConfig.valid_raw_data_file):
            valid_data = load_pkl(DatasetConfig.valid_raw_data_file)
            self.valid_dialogs = valid_data
        if self.mode & TEST_MODE and isfile(DatasetConfig.test_raw_data_file):
            test_data = load_pkl(DatasetConfig.test_raw_data_file)
            self.test_dialogs = test_data

    @staticmethod
    def _get_common_data() -> CommonData:
        """Get common data.

        Returns:
            * CommonData: Common data.

        """

        dialog_vocab = RawData._get_dialog_vocab()
        glove = RawData._get_glove(dialog_vocab)
        image_url_id, image_paths = RawData._get_images()
        return CommonData(dialog_vocab=dialog_vocab, glove=glove,
                          image_url_id=image_url_id,
                          image_paths=image_paths)

    @staticmethod
    def _get_dialog_vocab() -> Dict[str, int]:
        """Get dialog vocabulary.

        Returns:
            Dict[str, int]

        """
        word_freq_cnt = Counter()
        RawData._process_dialog_dir(DatasetConfig.train_dialog_data_directory,
                                    word_freq_cnt=word_freq_cnt)
        RawData._process_dialog_dir(DatasetConfig.valid_dialog_data_directory,
                                    word_freq_cnt=word_freq_cnt)
        words = copy.copy(SPECIAL_TOKENS)
        words += [word for word, freq in word_freq_cnt.most_common()
                  if freq >= DatasetConfig.dialog_text_cutoff]
        vocab: Dict[str, int] = {word: wid for wid, word in enumerate(words)}
        return vocab

    @staticmethod
    def _get_glove(vocab: Dict[str, int]) -> List[Optional[List[float]]]:
        """Get GloVe (Global Vectors for Word Representation)

        Args:
            vocab (Dict[str, int]): Vocabulary.

        Returns:
            List[Optional[List[float]]]: Extracted GloVe, which each element
            in the list is either a float vector or None (no such word in
            GloVe file). Element of index i is the GloVe of ith word (
            corresponding to vocab).

        """
        # Read raw Glove file.
        print('Reading GloVe file {}...'.format(DatasetConfig.glove_file))
        with open(DatasetConfig.glove_file, 'r') as file:
            raw_glove: Dict[str, List[float]] = {}
            for line in file:
                line = line.strip().split(' ')
                if line:
                    raw_glove[line[0]] = list(map(float, line[1:]))

        # Extract needed vectors.
        glove: List[Optional[List[float]]] = [None] * len(vocab)
        for word, idx in vocab.items():
            if idx >= len(SPECIAL_TOKENS):
                glove[idx] = raw_glove.get(word, None)
        return glove

    @staticmethod
    def _process_dialog_dir(dialog_dir: str, vocab: Dict[str, int] = None,
                            image_url_id: Dict[str, int] = None,
                            word_freq_cnt: Counter = None) -> List[Dialog]:
        """Process dialog directory.

        Args:
            dialog_dir (str): Dialog directory.
            vocab (Dict[str, int], optional): Vocabulary.
            image_url_id (Dict[str, int], optional): Image URL to index.
            word_freq_cnt (Counter, optional): Word frequency counter.

        Note:
            * (word_freq_cnt is not None) or (vocab is not None and
              image_url_id is not None) == True
            * word_freq_cnt will be updated if it's not None

        Returns:
            List[Dialog]: Extracted dialogs. Empty list if
            word_freq_cnt is not None.

        """

        # Check arguments.
        assert (word_freq_cnt is not None) or (vocab is not None and
                                               image_url_id is not None)

        get_vocab: bool = word_freq_cnt is not None

        print('Processing dialog directory {}...'.format(dialog_dir))
        files = listdir(dialog_dir)

         # Useless if word_freq_cnt is not None.
        dialogs: List[Dialog] = []

        for file_idx, file in enumerate(files):
            if file.endswith('.json'):
                full_path = join(dialog_dir, file)

                # Print current progress.
                #if (file_idx + 1) % DIALOG_PROC_PRINT_FREQ == 0:
                #    print('Processing dialog directory: {}/{}'.format(
                #        file_idx + 1, len(files)))

                # Load JSON.
                try:
                    dialog_json = json.load(open(full_path))
                except json.decoder.JSONDecodeError:
                    continue
                    
                for dial_idx in range(len(dialog_json['dialogue_data'])):
                    # Extract useful information
                    dialog = []
                    dial = dialog_json['dialogue_data'][dial_idx]['dialogue']
                    for dial_idx2 in range(len(dial)):
                        #dial_data = dial[dial_idx2]
                        utter_dict = dial[dial_idx2]#dial_data['dialogue']
                        utter_coref = dialog_json['dialogue_data'][dial_idx]['dialogue_coref_map']
                        if not get_vocab:
                            user_utter = RawData._get_utter_from_dict(vocab,
                                                                 image_url_id,
                                                                 utter_dict,
                                                                 utter_coref,
                                                                 speaker = 'user')
                            dialog.append(user_utter)
                            sys_utter = RawData._get_utter_from_dict(vocab,
                                                                 image_url_id,
                                                                 utter_dict,
                                                                 utter_coref,
                                                                 speaker = 'sys')
                            dialog.append(sys_utter)
                        else:
                            # Collect vocab from system transcript and user transcript
                            print("Collecting vocab from dialogues...")
                            for transcript_source in ["transcript", 'system_transcript']:
                                text: str = utter_dict.get(transcript_source)
                                if text is None:
                                    text = ''
                                words: List[str] = word_tokenize(text)
                                words = [word.lower() for word in words]
                                if get_vocab:
                                    word_freq_cnt.update(words)
                    if not get_vocab:
                        dialogs.append(dialog)
        
        return dialogs

    @staticmethod
    def _get_utter_from_dict(vocab: Dict[str, int],
                             image_url_id: Dict[str, int],
                             utter_dict: dict,
                             utter_coref: dict,
                             speaker: str) -> Utterance:
        """Extract Utterance object from JSON dict.

        Args:
            vocab (Dict[str, int]): Vocabulary.
            image_url_id (Dict[str, int]): Image URL to index.
            utter_dict (dict): JSON dict.

        Returns:
            Utterance: Extracted Utterance.

        """
        if speaker == 'sys':
            _speaker: str = 'system'
            _utter_type: str = (ast.literal_eval(utter_dict.get('system_transcript_annotated')))[0]['intent'].split(':')[0]
            _text: str = utter_dict['system_transcript']
        if speaker == 'user':
            _speaker: str = 'user'
            _utter_type: str = (ast.literal_eval(utter_dict.get('system_transcript_annotated')))[0]['intent'].split(':')[0]
            _text: str = utter_dict['transcript']
        _pos_images: List[str] = [] #list(utter_coref.keys())
        _neg_images: List[str] = []

        # Some attributes may be empty.
        if _text is None:
            _text = ""
        if _utter_type is None:
            _utter_type = ""
        if _pos_images is None:
            _pos_images = []
        if _neg_images is None:
            _neg_images = []

        # Convert speaker into an integer.
        speaker: int = -1
        if _speaker == 'user':
            speaker = USER_SPEAKER
        elif _speaker == 'system':
            speaker = SYS_SPEAKER
        assert speaker != -1

        # Convert utterance type into an integer.
        utter_type: int = DatasetConfig.utterance_type_dict.get(_utter_type, 0)
        # We don't care the type of system response.
        if speaker == SYS_SPEAKER:
            utter_type = 0

        # Convert text into a list of integers.
        words: List[str] = word_tokenize(_text)
        text: List[int] = [vocab.get(word.lower(), UNK_ID) for word in words]

        # Images
        #pos_images: List[int] = [image_url_id.get(img, 0)
        #                         for img in _pos_images]
        pos_images: List[str] = _pos_images
        neg_images: List[int] = [image_url_id.get(img, 0)
                                 for img in _neg_images]

        utter = Utterance(speaker, utter_type, text, pos_images, neg_images)
        return utter

    @staticmethod
    def _get_images() -> Tuple[Dict[str, int], List[str]]:
        """Get images (URL and filenames of local images mapping).

        URL -> Path => URL -> index & index -> Path

        Returns:
            Dict[str, int]: Image URL to index.
            List[str]: Index to the filename of the local image.

        """

        # Get URL to filename mapping dict.
        with open(DatasetConfig.url2img, 'r') as file:
            url_image_pairs: List[List[str]] = [line.strip().split(' ')
                                                for line in file.readlines()]
        url_image_pairs: List[Tuple[str, str]] = [(p[0], p[1])
                                                  for p in url_image_pairs]
        url2img: Dict[str, str] = dict(url_image_pairs)

        # Divided it into two steps.
        # URL -> Path => URL -> index & index -> Pathtrain
        # Element of index 0 should be empty image.
        image_url_id: Dict[str, int] = {'': 0}
        image_paths: List[str] = ['']

        for url, img in url2img.items():
            image_url_id[url] = len(image_url_id)
            image_paths.append(img)
        return image_url_id, image_paths

    @staticmethod
    def _get_dialogs(mode: int,
                     vocab: Dict[str, int],
                     image_url_id: Dict[str, int]) -> List[Dialog]:
        """Get mode specific dialogs.

        Args:
            mode (int): TRAIN_MODE / VALID_MODE / TEST_MODE.
            vocab (Dict[str, int]): Vocabulary.
            image_url_id (Dict[str, int]): Image URL to index.

        Returns:
            List[Dialog]: Extracted dialogs of specific mode.

        Raises:
            ValueError: Raises if mode is neither TRAIN_MODE, nor VALID_MODE,
                        nor TEST_MODE.

        """

        if mode == TRAIN_MODE:
            return RawData._process_dialog_dir(
                DatasetConfig.train_dialog_data_directory, vocab, image_url_id)
        if mode == VALID_MODE:
            return RawData._process_dialog_dir(
                DatasetConfig.valid_dialog_data_directory, vocab, image_url_id)
        if mode == TEST_MODE:
            return RawData._process_dialog_dir(
                DatasetConfig.test_dialog_data_directory, vocab, image_url_id)
        raise ValueError('Illegal mode.')
