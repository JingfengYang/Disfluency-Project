from utils import readData,preProcess,build_vocab,batchIter,idData,isEdit
import torch
import torch.nn as nn
import time
from torch import optim
import torch.nn.functional as F

WORD_EMBEDDING_DIM = 64
INPUT_DIM = 100
HIDDEN_DIM = 256
PRINT_EVERY=100
EVALUATE_EVERY_EPOCH=1
LEARNING_RATE=0.0
ENCODER_LAYER=2
DROUPOUT_RATE=0.1
BATCH_SIZE=32
INIT_LEARNING_RATE=0.0001
EPOCH=200

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
        self.tanh1 = nn.ReLU()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=self.nLayers, bidirectional=True)
        self.hidden2output1 = nn.Linear(hidden_dim*2,hidden_dim)
        self.tanh2 = nn.ReLU()
        self.hidden2output2 = nn.Linear(hidden_dim, labelSize)

        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self,input,lengths, train=True,hidden=None):
        input=self.tanh1(self.embeds2input(self.word_embeds(input)))
        if train:
            input=self.dropout1(input)
        input=input.transpose(0,1)
        input=nn.utils.rnn.pack_padded_sequence(input, lengths)
        output, hidden = self.lstm(input, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        if train:
            output=self.dropout2(output)
        output=output.transpose(0,1)
        output=self.hidden2output2(self.tanh2(self.hidden2output1(output)))
        output=F.log_softmax(output, dim=-1)
        return output


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self,  criterion, opt=None):
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm,train=True):
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        if train:
            loss.backward()
            if self.opt is not None:
                #print(loss)
                self.opt.step()
                self.opt.optimizer.zero_grad()
        else:
            if self.opt is not None:
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
        true_dist = x.clone().detach()
        true_dist.fill_(self.smoothing / (self.size - 1))

        true_dist.scatter_(1, target.masked_fill(target == self.padding_idx,0).unsqueeze(1), self.confidence)
        mask = torch.nonzero(target == self.padding_idx)
        #print(mask.squeeze().size(),true_dist.size())
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)

def run_epoch(data_iter, model, loss_compute,train=True,id2label=None):

    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    editTrueTotal = 0
    editPredTotal = 0
    editCorrectTotal = 0
    for i, (sent_batch,tag_batch) in enumerate(data_iter):##
        out = model(sent_batch[0], sent_batch[1],train=train)
        loss = loss_compute(out, tag_batch[0], sent_batch[2],train=train)
        total_loss += loss
        total_tokens += sent_batch[2]
        tokens += sent_batch[2]

        if i %  PRINT_EVERY== 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Tokens per Sec: %f Loss: %f " %
                    (i, tokens / elapsed , loss / sent_batch[2] ))
            start = time.time()
            tokens = 0
        if not train:
            pad=out.size(-1)
            _, results = torch.max(out.contiguous().view(-1, out.size(-1)), 1)
            results = results.detach().tolist()
            y = tag_batch[0].contiguous().view(-1).detach().tolist()

            for pred,gold in zip(results,y):
                if not gold==pad:
                    if isEdit(pred,id2label) and isEdit(gold,id2label):
                        editCorrectTotal+=1
                        editTrueTotal+=1
                        editPredTotal+=1
                    else:
                        if isEdit(pred,id2label):
                            editPredTotal += 1
                        if isEdit(gold,id2label):
                            editTrueTotal +=1
    f=0.0
    if not train:
        if not editPredTotal:
            editPredTotal = 1
            editCorrectTotal=1
        if not editCorrectTotal:
            editCorrectTotal=1
        p = editCorrectTotal / editPredTotal
        r = editCorrectTotal / editTrueTotal
        f=2 * p * r / (p + r)
        print("Edit word precision: %f recall: %f fscore: %f" % (p, r, f))

    return total_loss / total_tokens,f


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

def run(epoch,model,batch_size,trainData,valData,testData,id2label):
    valResult=[]
    testResult=[]
    criterion = LabelSmoothing(size=len(id2label), padding_idx=len(id2label), smoothing=0.0)
    model_opt = NoamOpt(HIDDEN_DIM, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE,betas=(0.9, 0.98), eps=1e-9))
    for i in range(epoch):
        model.train()
        run_epoch(batchIter(trainData,batch_size,tag_pad=len(id2label)), model,
                  SimpleLossCompute( criterion, model_opt),train=True)
        model.eval()
        print('Evaluation_val: epoch: %d' % (i))
        loss,f=run_epoch(batchIter(testData, batch_size, tag_pad=len(id2label)), model,
                  SimpleLossCompute(criterion, model_opt), train=False, id2label=id2label)
        print('Loss:', loss)
        valResult.append(f)
        print('Evaluation_test: epoch: %d' % (i))
        loss,f=run_epoch(batchIter(valData, batch_size, tag_pad=len(id2label)), model,
                                           SimpleLossCompute(criterion, model_opt), train=False, id2label=id2label)
        print('Loss:', loss)
        testResult.append(f)
    valBest=max(valResult)
    print('ValBest epoch:', [i for i, j in enumerate(valResult) if j == valBest])
    testBest = max(testResult)
    print('TestBest epoch:', [i for i, j in enumerate(testResult) if j == testBest])


trainSents=preProcess(readData('train'))
valSents=preProcess(readData('val'))
testSents=preProcess(readData('test'))
vocDic,label2id,id2label=build_vocab(trainSents)
print(id2label)
encoder=Encoder(len(vocDic),WORD_EMBEDDING_DIM,INPUT_DIM,HIDDEN_DIM,ENCODER_LAYER,len(id2label),DROUPOUT_RATE).to(device)
trainData=idData(trainSents,vocDic,label2id)
valData=idData(valSents,vocDic,label2id)
testData=idData(testSents,vocDic,label2id)
run(EPOCH,encoder,BATCH_SIZE,trainData,valData,testData,id2label)