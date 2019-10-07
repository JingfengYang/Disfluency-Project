from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from typing import Iterator, List, Dict
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers import Token
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField


class DisfluencyDatasetReader(DatasetReader):
    """
    DatasetReader for Disfluency tagging data
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token],
                         label: str = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if label:
            label_field = LabelField(label)
            fields["label"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            state = 0
            tokens = []
            label = None
            for line in f:
                if line.strip() == '':
                    yield self.text_to_instance(
                        [Token(word) for word in tokens],
                        label)
                    continue
                elif state == 0:
                    tokens = line.strip().split()
                    state = 1
                    continue
                elif state == 1:
                    # POS tags, discard
                    state = 2
                    continue
                elif state == 2:
                    tags = line.strip().split()
                    if tags.count('O') + tags.count('OR') == len(tags):
                        label = "fluent"
                    else:
                        label = "disfluent"
                    state = 0
