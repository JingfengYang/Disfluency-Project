import torch
import torch.nn as nn
import torch.optim as optim

from allennlp.data.token_indexers import PretrainedBertIndexer

from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder \
    import PretrainedBertEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.training.learning_rate_schedulers.noam import NoamLR

from allennlp.data.iterators import BucketIterator, BasicIterator

from allennlp.training.trainer import Trainer
from allennlp.training.util import evaluate

from BiLSTMClassifier import BiLSTMClassifier
from DisfluencyDatasetReader import DisfluencyDatasetReader
from DisfluencyPredictor import DisfluencyPredictor as Predictor


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
    EPOCH = 30
    WARMUP_STEPS = 10000
    PATIENCE = 5

    torch.manual_seed(1)

    token_indexer = PretrainedBertIndexer(
        pretrained_model="bert-base-uncased",
        do_lowercase=True
    )
    reader = DisfluencyDatasetReader(
        token_indexers={"tokens": token_indexer})

    train_dataset = reader.read('../train.txt')
    validation_dataset = reader.read('../val.txt')
    # test_dataset = reader.read('../test.txt')
    test_dataset = reader.read('../skysports_extracted.txt')

    for instance in test_dataset:
        print(instance)
    print(len(test_dataset))

    vocab = Vocabulary.from_instances(
        train_dataset + validation_dataset + test_dataset)

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=WORD_EMBEDDING_DIM)
    bert_embedder = PretrainedBertEmbedder(
        pretrained_model="bert-base-uncased",
        requires_grad=False,
        top_layer_only=True
    )
    word_embeddings = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                             allow_unmatched_keys=True)

    lstm = PytorchSeq2VecWrapper(nn.LSTM(
        INPUT_DIM, HIDDEN_DIM,
        num_layers=ENCODER_LAYER,
        bidirectional=True,
        batch_first=True))

    if torch.cuda.is_available():
        cuda_device = 0
    else:
        cuda_device = -1

    # # And here's how to reload the model.
    # vocab2 = Vocabulary.from_files("vocabulary")
    model = BiLSTMClassifier(word_embeddings, lstm, DROPOUT_RATE, vocab)
    with open("model.th", 'rb') as f:
        model.load_state_dict(torch.load(f))
    if cuda_device > -1:
        model.cuda(cuda_device)

    seq_iterator = BasicIterator(batch_size=32)
    seq_iterator.index_with(vocab)

    predictor = Predictor(model, seq_iterator, vocab, cuda_device=cuda_device)
    test_preds = predictor.predict(test_dataset)

    disfluent = []

    with open('predictions.txt', 'w+') as f:
        for i in range(len(test_dataset)):
            sen = " ".join([str(x) for x in
                            test_dataset[i].__getitem__('sentence').tokens])
            f.write("{}\n".format(sen))
            f.write("{}\n".format(test_preds[i]))
            if test_preds[i] == 'disfluent':
                disfluent.append(sen)

    with open('disfluent.txt', 'w+') as f:
        for i in range(len(disfluent)):
            f.write("{}\n".format(disfluent[i]))
