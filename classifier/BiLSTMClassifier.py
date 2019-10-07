import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterator, List, Dict
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules import FeedForward
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.models import Model
from allennlp.nn.activations import Activation
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask, \
    sequence_cross_entropy_with_logits, masked_log_softmax


class BiLSTMClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
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

        self.encoder = encoder

        self.hidden2intermediate = FeedForward(
            input_dim=encoder.get_output_dim(),
            num_layers=1,
            hidden_dims=int(encoder.get_output_dim() / 2),
            activations=Activation.by_name('relu')(),
            dropout=dropout_p)

        self.intermediate2tag = nn.Linear(
            in_features=int(encoder.get_output_dim() / 2),
            out_features=vocab.get_vocab_size('labels'))

        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_in = self.embedding2input(embeddings)
        encoder_out = self.encoder(encoder_in, mask)
        intermediate = self.hidden2intermediate(encoder_out)
        tag_logits = self.intermediate2tag(intermediate)
        probs = F.softmax(tag_logits, dim=-1)
        output = {"tag_logits": tag_logits, 'probs': probs}
        if label is not None:
            self.accuracy(tag_logits, label)
            output["loss"] = self.loss_function(tag_logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # return {"accuracy": self.accuracy.get_metric(reset)}
        metric = self.accuracy.get_metric(reset)
        return {"accuracy": metric}
