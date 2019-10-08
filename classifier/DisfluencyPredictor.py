from allennlp.data.iterators import DataIterator
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from tqdm import tqdm
import numpy as np
from typing import Iterable
from allennlp.data import Instance
from allennlp.nn.util import move_to_device
from scipy.special import expit  # the sigmoid function
import torch


def to_np(tsr):
    return tsr.detach().cpu().numpy()


class DisfluencyPredictor:
    def __init__(self,
                 model: Model,
                 iterator: DataIterator,
                 vocab: Vocabulary,
                 cuda_device: int=-1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device
        self.vocab = vocab

    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch)
        softmax_probs = expit(to_np(out_dict["tag_logits"]))
        argmax_indices = np.argmax(softmax_probs, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        return labels

    def predict(self, ds: Iterable[Instance]) -> np.ndarray:
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        preds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = move_to_device(batch, self.cuda_device)
                preds += self._extract_data(batch)

        return preds
