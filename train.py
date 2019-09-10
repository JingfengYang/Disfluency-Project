from utils import readData,preProcess,build_vocab,batchIter,idData
import torch
import torch.nn as nn
import time
from torch import optim
import torch.nn.functional as F

WORD_EMBEDDING_DIM = 64
INPUT_DIM = 100
HIDDEN_DIM = 256
PRINT_EVERY=10
EVALUATE_EVERY_EPOCH=1
LEARNING_RATE=0.0005
ENCODER_LAYER=2
DROUPOUT_RATE=0.1
BATCH_SIZE=20
EPOCH=10

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self,word_size,word_dim,input_dim,hidden_dim,nLayers,labelSize,dropout_p):
        super(Encoder, self).__init__()
        self.nLayers=nLayers
        self.hidden_dim=hidden_dim
        self.input_dim=input_dim

        self.word_embeds = nn.Embedding(word_size, word_dim)
        self.embeds2input = nn.Linear(word_dim, self.input_dim)
        self.relu1 = nn.ReLU()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=self.nLayers, bidirectional=True)
        self.hidden2output1 = nn.Linear(hidden_dim*2,hidden_dim)
        self.relu2 = nn.ReLU()
        self.hidden2output2 = nn.Linear(hidden_dim, labelSize)

        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self,input,lengths, train=True,hidden=None):
        input=self.relu(self.embeds2input(self.word_embeds(input)))
        if train:
            input=self.dropout1(input)
        input=input.transpose(0,1)
        input=nn.utils.rnn.pack_padded_sequence(input, lengths)
        output, hidden = self.lstm(input, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        if train:
            output=self.dropout2(output)
        output=output.transpose(0,1)
        output=self.hidden2output2(self.relu2(self.hidden2output1(output)))
        return output


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self,  criterion, opt=None):
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx=padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        mask = torch.nonzero(target == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, torch.tensor(true_dist, requires_grad=False,dtype=torch.long, device=device))

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, (sent_batch,tag_batch) in enumerate(data_iter):##
        out = model(sent_batch[0], sent_batch[1])
        loss = loss_compute(out, tag_batch[0], sent_batch[2])
        total_loss += loss
        total_tokens += sent_batch[2]
        tokens += sent_batch[2]
        if i %  PRINT_EVERY== 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / sent_batch[2], tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

def train(epoch,model,batch_size,trainData,valData,testData,id2label):
    criterion = LabelSmoothing(size=len(id2label), padding_idx=0, smoothing=0.0)
    model_opt = NoamOpt(HIDDEN_DIM, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(epoch):
        model.train()
        run_epoch(batchIter(trainData,batch_size), model,
                  SimpleLossCompute( criterion, model_opt))
        model.eval()
        print(run_epoch(batchIter(trainData,batch_size), model,
                  SimpleLossCompute( criterion, model_opt)))

trainSents=preProcess(readData('train'))
valSents=preProcess(readData('val'))
testSents=preProcess(readData('test'))
vocDic,label2id,id2label=build_vocab(trainSents)
encoder=Encoder(len(vocDic),WORD_EMBEDDING_DIM,INPUT_DIM,HIDDEN_DIM,ENCODER_LAYER,len(id2label),DROUPOUT_RATE).to(device)
trainData=idData(trainSents,vocDic,label2id)
valData=idData(valSents,vocDic,label2id)
testData=idData(testSents,vocDic,label2id)
train(EPOCH,encoder,BATCH_SIZE,trainData,valData,testData,id2label)


