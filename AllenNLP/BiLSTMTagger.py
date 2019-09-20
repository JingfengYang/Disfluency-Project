import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterator, List, Dict
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules import FeedForward
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from metric import CustomFBetaMeasure
from allennlp.models import Model
from allennlp.nn.activations import Activation
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask, \
    sequence_cross_entropy_with_logits, masked_log_softmax


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

        # self.accuracy = CategoricalAccuracy()

        label_vocab = vocab.get_token_to_index_vocabulary('labels').copy()
        # print("label_vocab: ", label_vocab)
        [label_vocab.pop(x) for x in ['O', 'OR']]
        labels_for_metric = list(label_vocab.values())
        # print("labels_for_metric: ", labels_for_metric)
        self.accuracy = CustomFBetaMeasure(beta=1.0,
                                           average='micro',
                                           labels=labels_for_metric)

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_in = self.embedding2input(embeddings)
        encoder_out = self.encoder(encoder_in, mask)
        intermediate = self.hidden2intermediate(encoder_out)
        tag_logits = self.intermediate2tag(intermediate)
        probs = F.softmax(tag_logits, dim=-1)
        output = {"tag_logits": tag_logits, 'probs': probs}
        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(
                tag_logits, labels, mask)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # return {"accuracy": self.accuracy.get_metric(reset)}
        metric = self.accuracy.get_metric(reset)
        return {"precision": metric['precision'],
                "recall": metric['recall'],
                "fscore": metric['fscore']}
