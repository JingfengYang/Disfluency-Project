import torch
import torch.nn as nn
import torch.optim as optim

from allennlp.data.token_indexers import PretrainedBertIndexer

from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder \
    import PretrainedBertEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.training.learning_rate_schedulers.noam import NoamLR

from allennlp.data.iterators import BucketIterator, BasicIterator

from allennlp.training.trainer import Trainer
from allennlp.training.util import evaluate

from BiLSTMTagger import BiLSTMTagger
from DisfluencyDatasetReader import DisfluencyDatasetReader
from DisfluencyPredictor import DisfluencyPredictor


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
    EPOCH = 1000
    WARMUP_STEPS = 5000
    PATIENCE = 10
    BERT = False

    torch.manual_seed(1)

    if BERT:
        token_indexer = PretrainedBertIndexer(
            pretrained_model="bert-base-uncased",
            do_lowercase=True
        )
        reader = DisfluencyDatasetReader(
            token_indexers={"tokens": token_indexer})
    else:
        reader = DisfluencyDatasetReader()

    train_dataset = reader.read('../train.txt')
    validation_dataset = reader.read('../val.txt')
    test_dataset = reader.read('../test.txt')

    vocab = Vocabulary.from_instances(
        train_dataset + validation_dataset + test_dataset)

    if BERT:
        bert_embedder = PretrainedBertEmbedder(
            pretrained_model="bert-base-uncased",
            requires_grad=False,
            top_layer_only=True
        )
        word_embeddings = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                                 allow_unmatched_keys=True)
    else:

        token_embedding = Embedding(
            num_embeddings=vocab.get_vocab_size('tokens'),
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

    # optimizer = optim.SGD(model.parameters(), lr=0.1)

    optimizer = optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE,
                           betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamLR(
        optimizer=optimizer,
        model_size=HIDDEN_DIM,
        warmup_steps=WARMUP_STEPS,
        factor=1)

    iterator = BucketIterator(batch_size=BATCH_SIZE, sorting_keys=[
                              ("sentence", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      learning_rate_scheduler=scheduler,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=PATIENCE,
                      num_epochs=EPOCH,
                      cuda_device=cuda_device)

    trainer.train()

    # Here's how to save the model.
    with open("model.th", 'wb') as f:
        torch.save(model.state_dict(), f)
    vocab.save_to_files("vocabulary")

    # # And here's how to reload the model.
    # vocab2 = Vocabulary.from_files("vocabulary")
    # model2 = BiLSTMTagger(word_embeddings, lstm, vocab2)
    # with open("model.th", 'rb') as f:
    #     model2.load_state_dict(torch.load(f))
    # if cuda_device > -1:
    #     model2.cuda(cuda_device)

    seq_iterator = BasicIterator(batch_size=32)
    seq_iterator.index_with(vocab)

    metrics = evaluate(model=model,
                       instances=test_dataset,
                       data_iterator=seq_iterator,
                       cuda_device=cuda_device,
                       batch_weight_key=None)
    print("Test accuracy: ", metrics)
