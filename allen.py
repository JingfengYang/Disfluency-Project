from typing import Iterator, List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField

from allennlp.data.dataset_readers import DatasetReader

from allennlp.common.file_utils import cached_path

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder, \
    BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, \
    PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, \
    sequence_cross_entropy_with_logits, masked_log_softmax
from allennlp.nn.activations import Activation

from allennlp.training.metrics import CategoricalAccuracy

from allennlp.data.iterators import BucketIterator

from allennlp.training.trainer import Trainer

from allennlp.predictors import SentenceTaggerPredictor
torch.manual_seed(1)


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
                         tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = SequenceLabelField(
                labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            state = 0
            tokens = []
            tags = []
            for line in f:
                if line.strip() == '':
                    yield self.text_to_instance(
                        [Token(word) for word in tokens],
                        tags)
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
                    state = 0


class BiLSTMTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 dropout_p: int,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)

        self.word_embeddings = word_embeddings

        self.embedding2input = FeedForward(
            input_dim=word_embeddings.get_output_dim(),
            num_layers=1,
            hidden_dims=encoder.get_input_dim(),
            activations=Activation.by_name('relu')(),
            dropout=dropout_p)

        # self.embedding2input = nn.Linear(
        #     in_features=word_embeddings.get_output_dim(),
        #     out_features=encoder.get_input_dim())
        # self.relu1 = nn.ReLU()

        self.encoder = encoder

        self.hidden2intermediate = FeedForward(
            input_dim=encoder.get_output_dim(),
            num_layers=1,
            hidden_dims=int(encoder.get_output_dim() / 2),
            activations=Activation.by_name('relu')(),
            dropout=dropout_p)

        # self.hidden2intermediate = nn.Linear(
        #     in_features=encoder.get_output_dim(),
        #     out_features=int(encoder.get_output_dim() / 2))
        # self.relu2 = nn.ReLU()

        self.intermediate2tag = nn.Linear(
            in_features=int(encoder.get_output_dim() / 2),
            out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()

        # self.dropout1 = nn.Dropout(dropout_p)
        # self.dropout2 = nn.Dropout(dropout_p)

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_in = self.embedding2input(embeddings)
        encoder_out = self.encoder(encoder_in, mask)
        intermediate = self.hidden2intermediate(encoder_out)
        tag_logits = self.intermediate2tag(intermediate)
        # probs = masked_log_softmax(result, mask)
        output = {"tag_logits": tag_logits}
        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(
                tag_logits, labels, mask)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


if __name__ == '__main__':
    WORD_EMBEDDING_DIM = 64
    INPUT_DIM = 100
    HIDDEN_DIM = 256
    PRINT_EVERY = 1000
    EVALUATE_EVERY_EPOCH = 1
    ENCODER_LAYER = 2
    DROPOUT_RATE = 0.1
    BATCH_SIZE = 32
    INIT_LEARNING_RATE = 0.0
    EPOCH = 400

    reader = DisfluencyDatasetReader()

    train_dataset = reader.read('train.txt')
    validation_dataset = reader.read('val.txt')
    test_dataset = reader.read('test.txt')

    vocab = Vocabulary.from_instances(
        train_dataset + validation_dataset + test_dataset)

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=WORD_EMBEDDING_DIM)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    lstm = PytorchSeq2SeqWrapper(nn.LSTM(
        INPUT_DIM, HIDDEN_DIM,
        num_layers=ENCODER_LAYER,
        bidirectional=True,
        batch_first=True))

    model = BiLSTMTagger(word_embeddings, lstm, DROPOUT_RATE, vocab)

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    iterator = BucketIterator(batch_size=BATCH_SIZE, sorting_keys=[
                              ("sentence", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=10,
                      num_epochs=1000,
                      cuda_device=cuda_device)

    trainer.train()

    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)

    tag_logits = predictor.predict("The dog ate the apple")['tag_logits']

    tag_ids = np.argmax(tag_logits, axis=-1)

    print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

    # Here's how to save the model.
    with open("/tmp/model.th", 'wb') as f:
        torch.save(model.state_dict(), f)

    vocab.save_to_files("/tmp/vocabulary")

    # And here's how to reload the model.
    vocab2 = Vocabulary.from_files("/tmp/vocabulary")

    model2 = LstmTagger(word_embeddings, lstm, vocab2)

    with open("/tmp/model.th", 'rb') as f:
        model2.load_state_dict(torch.load(f))

    if cuda_device > -1:
        model2.cuda(cuda_device)

    predictor2 = SentenceTaggerPredictor(model2, dataset_reader=reader)
    tag_logits2 = predictor2.predict("The dog ate the apple")['tag_logits']
    np.testing.assert_array_almost_equal(tag_logits2, tag_logits)
