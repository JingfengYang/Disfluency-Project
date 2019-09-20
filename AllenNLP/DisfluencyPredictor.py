from allennlp.data.iterators import DataIterator
from allennlp.models import Model
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
                 cuda_device: int=-1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device

    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch)
        return expit(to_np(out_dict["tag_logits"]))

    def predict(self, ds: Iterable[Instance]) -> np.ndarray:
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        preds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = move_to_device(batch, self.cuda_device)
                preds.append(self._extract_data(batch))
        return np.concatenate(preds, axis=0)
