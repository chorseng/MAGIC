import json
from collections import Counter
from os import listdir
from os.path import isfile, join
from typing import List, Dict, Tuple

from config import DatasetConfig
from util import load_pkl, save_pkl


class StyleTipsData:
    def __init__(self, vocab=None, edges=None, scores=None):
        # Attributes.
        self.vocab: Dict[str, int] = vocab
        self.edges: List[Tuple[int, int]] = edges
        self.scores: List[float] = scores

    def __repr__(self):
        res = ''
        res += 'vocab: ' + self.vocab.__repr__() + '\n'
        res += 'edges: ' + self.edges.__repr__() + '\n'
        res += 'scores: ' + self.scores.__repr__() + '\n'
        return res

    @staticmethod
    def from_file():
        """Build a StyleTipsData object from file."""
        if not isfile(DatasetConfig.styletips_data_file):
            raise ValueError('No style tips data file.')

        vocab: Dict[str, int] = {}
        edges: List[Tuple[int, int]] = []
        scores: List[float] = []

        with open(DatasetConfig.styletips_data_file) as file:
            for line in file:
                products: List[str] = [None] * 2
                products[0], products[1], score = map(lambda x: x.strip(),
                                                      line.split(','))
                products = list(map(lambda x: x.lower(), products))
                score = int(score)
                for product in products:
                    if product not in vocab:
                        vocab[product] = len(vocab)
                edges.append((vocab[products[0]], vocab[products[1]]))
                scores.append(score)

        return StyleTipsData(vocab, edges, scores)


class CelebrityData:
    def __init__(self, celebrity_id=None, product_id=None, scores=None):
        self.celebrity_id: Dict[str, int] = celebrity_id
        self.product_id: Dict[str, int] = product_id
        self.scores = scores

    def __repr__(self):
        res = ''
        res += 'celebrity_id: ' + self.celebrity_id.__repr__() + '\n'
        res += 'product_id: ' + self.product_id.__repr__() + '\n'
        res += 'scores: ' + self.scores.__repr__() + '\n'
        return res

    @staticmethod
    def from_file():
        """Build a CelebrityData object from file."""
        if not isfile(DatasetConfig.celebrity_data_file):
            raise ValueError('No celebrity data file.')

        celebrity_id: Dict[str, int] = {}
        product_id: Dict[str, int] = {}

        # Load JSON file.
        with open(DatasetConfig.celebrity_data_file) as file:
            celebrity_json = json.load(file)

        # Count celebrities and product types.
        for celebrity, products in celebrity_json.items():
            celebrity = celebrity.lower()
            # Assign ids for new celebrity.
            if celebrity not in celebrity_id:
                celebrity_id[celebrity] = len(celebrity_id)
            for product in products.keys():
                product = product.lower()
                if product not in product_id:
                    product_id[product] = len(product_id)

        scores = [[0] * len(product_id) for _ in range(len(celebrity_id))]
        for celebrity, products in celebrity_json.items():
            celebrity = celebrity.lower()
            cel_id = celebrity_id[celebrity]
            for product, score in products.items():
                product = product.lower()
                prod_id = product_id[product]
                scores[cel_id][prod_id] = score

        return CelebrityData(celebrity_id, product_id, scores)


class AttributeData:
    def __init__(self, key_vocab=None, value_vocab=None):
        self.key_vocab: Dict[str, int] = key_vocab
        self.value_vocab: Dict[Tuple[int, str], int] = value_vocab

    def __repr__(self):
        res = ''
        res += 'key_vocab: ' + self.key_vocab.__repr__() + '\n'
        res += 'value_vocab: ' + self.value_vocab.__repr__() + '\n'
        return res

    @staticmethod
    def from_file():
        key_vocab = {attr: idx for idx, attr in
                     enumerate(DatasetConfig.product_attributes)}
        value_vocab = {}

        # Count value of each attribute.
        counters: Dict[str, Counter] = {attr: Counter() for attr in
                                        DatasetConfig.product_attributes}
        print('Building attribute data from files...')
        files = listdir(DatasetConfig.product_data_directory)
        print('# Product files: {}'.format(len(files)))
        for idx, file_name in enumerate(files):
            if (idx + 1) % 1000 == 0:
                print('{} / {}'.format(idx + 1, len(files)))
            file_path = join(DatasetConfig.product_data_directory, file_name)
            with open(file_path, 'r') as file:
                product_json: Dict[str, str] = json.load(file)
            for key, value in product_json.items():
                if key in counters:
                    key = key.lower()
                    value = value.lower()
                    counters[key].update([value])

        # Assign an index for each value.
        for key, counter in counters.items():
            key_id = key_vocab[key]
            values = [word for word, freq in counter.most_common() if
                      freq >= DatasetConfig.product_value_cutoff]
            for value in values:
                value_vocab[(key_id, value)] = len(value_vocab)
            print("Key {}: {}".format(key, len(values)))

        return AttributeData(key_vocab, value_vocab)


class KnowledgeData:
    def __init__(self):
        self.styletips_data: StyleTipsData = None
        self.celebrity_data: CelebrityData = None
        self.attribute_data: AttributeData = None

        if isfile(DatasetConfig.knowledge_data_file):
            # Read existed extracted data files.
            knowledge_data = load_pkl(DatasetConfig.knowledge_data_file)
            self.styletips_data = knowledge_data.styletips_data
            self.celebrity_data = knowledge_data.celebrity_data
            self.attribute_data = knowledge_data.attribute_data
        else:
            # Load data from raw data file and save them into pkl.
            self.styletips_data = StyleTipsData.from_file()
            self.celebrity_data = CelebrityData.from_file()
            self.attribute_data = AttributeData.from_file()
            save_pkl(self, 'KnowledgeData', DatasetConfig.knowledge_data_file)

    def __repr__(self):
        res = ''
        res += 'styletips_data: ' + self.styletips_data.__repr__() + '\n'
        res += 'celebrity_data: ' + self.celebrity_data.__repr__() + '\n'
        res += 'attribute_data: ' + self.attribute_data.__repr__() + '\n'
        return res
