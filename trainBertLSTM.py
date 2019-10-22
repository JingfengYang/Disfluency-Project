from utilsBertLSTM import build_vocab,batchIter,idData,isEdit
from preprocessPTB import readData,preProcess
import torch
import torch.nn as nn
import time
from torch import optim
import torch.nn.functional as F
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AdamW, WarmupLinearSchedule
import torch.nn.utils.rnn as rnn_utils

WORD_EMBEDDING_DIM = 64
INPUT_DIM = 100
HIDDEN_DIM = 256
PRINT_EVERY=1000
EVALUATE_EVERY_EPOCH=1
ENCODER_LAYER=2
DROUPOUT_RATE=0.1
BATCH_SIZE=32
INIT_LEARNING_RATE=0.00005
EPOCH=30
WARM_UP_STEPS=0
ADAM_EPSILON=1e-8

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''class NMTModel(nn.Module):

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False):

        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)'''

class Encoder(nn.Module):
    #def __init__(self,word_size,word_dim,input_dim,hidden_dim,nLayers,labelSize,dropout_p):
    def __init__(self, input_dim,hidden_dim,nLayers,labelSize,dropout_p):

        super(Encoder, self).__init__()
        '''self.nLayers=nLayers
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
        self.dropout2 = nn.Dropout(dropout_p)'''
        self.nLayers = nLayers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.embeds2input = nn.Linear(768, self.input_dim)
        self.tanh1 = nn.ReLU()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=self.nLayers, bidirectional=True)
        self.hidden2output1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.tanh2 = nn.ReLU()
        self.hidden2output2 = nn.Linear(hidden_dim, labelSize)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)



    def forward(self,input,masks,lengths, train=True,hidden=None):
        '''input=self.tanh1(self.embeds2input(self.word_embeds(input)))
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
        output=F.log_softmax(output, dim=-1)'''
        encoded_layers, _ = self.bert(input)
        enc =  [layer[starts.nonzero().squeeze(1)]
                   for layer, starts in zip(encoded_layers,masks)]
        #lengths2=[a.size(0) for a in enc]
        #assert(lengths.tolist()==lengths2)
        enc=rnn_utils.pad_sequence(enc, batch_first=True, padding_value=0.0)
        input=self.tanh1(self.embeds2input(enc))
        if train:
            input = self.dropout1(input)
        input = input.transpose(0, 1)
        input = nn.utils.rnn.pack_padded_sequence(input, lengths)
        output, hidden = self.lstm(input, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        if train:
            output = self.dropout2(output)
        output = output.transpose(0, 1)
        output = self.hidden2output2(self.tanh2(self.hidden2output1(output)))
        output = F.log_softmax(output, dim=-1)

        return output


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self,  criterion, opt, scheduler):
        self.criterion = criterion
        self.opt = opt
        self.scheduler=scheduler

    def __call__(self, x, y, mask,norm,train=True):
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm

        if train:
            loss.backward()
            if self.opt is not None:
                #print(loss)
                self.opt.step()
                self.scheduler.step()

                self.opt.zero_grad()
        else:
            if self.opt is not None:
                self.opt.zero_grad()

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
        #print('true_dist1',true_dist)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        #print('x',x)
        #print('true_dist',true_dist)
        ret=self.criterion(x, true_dist)
        ##print('init_loss:',ret)
        return ret

def run_epoch(data_iter, model, loss_compute,train=True,id2label=None):

    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    editTrueTotal = 0
    editPredTotal = 0
    editCorrectTotal = 0
    for i, (sent_batch,tag_batch,head_batch,tag_mask_batch) in enumerate(data_iter):##
        #print(tag_batch[0])
        masks=tag_mask_batch[0].bool()
        out = model(sent_batch[0], head_batch[0], tag_batch[1],train=train)
        loss = loss_compute(out, tag_batch[0], masks, sent_batch[2],train=train)
        #hprint('batch_loss',loss)
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
            length=tag_mask_batch[1].contiguous().detach().tolist()
            #print('results:',results,y)
            for pred, gold in zip(results, y):
                if not gold == pad:
                    if isEdit(pred, id2label) and isEdit(gold, id2label):
                        editCorrectTotal += 1
                        editTrueTotal += 1
                        editPredTotal += 1
                    else:
                        if isEdit(pred, id2label):
                            editPredTotal += 1
                        if isEdit(gold, id2label):
                            editTrueTotal += 1

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

def run(epoch,model,batch_size,trainData,valData,testData,id2label,w_padding):
    valResult=[]
    testResult=[]
    #LabelSmoothing(size=len(id2label), padding_idx=len(id2label), smoothing=0.0)
    t_total=(len(trainData[0])//BATCH_SIZE+1)*EPOCH
    criterion = LabelSmoothing(size=len(id2label), padding_idx=len(id2label), smoothing=0.0)
    optimizer = AdamW(model.parameters(), lr=INIT_LEARNING_RATE, eps=ADAM_EPSILON)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=WARM_UP_STEPS, t_total=t_total)
    #model_opt = NoamOpt(HIDDEN_DIM, 1, WARM_UP_STEPS,
                        #torch.optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE,betas=(0.9, 0.98), eps=1e-9))
    for i in range(epoch):
        model.train()
        run_epoch(batchIter(trainData,batch_size,w_tag_pad=w_padding,t_tag_pad=len(id2label)), model,
                  SimpleLossCompute( criterion, optimizer,scheduler),train=True)
        model.eval()
        print('Evaluation_val: epoch: %d' % (i))
        loss,f=run_epoch(batchIter(valData, batch_size, w_tag_pad=w_padding,t_tag_pad=len(id2label)), model,
                  SimpleLossCompute(criterion, optimizer,scheduler), train=False, id2label=id2label)
        print('Loss:', loss)
        valResult.append(f)
        print('Evaluation_test: epoch: %d' % (i))
        loss,f=run_epoch(batchIter(testData, batch_size, w_tag_pad=w_padding,t_tag_pad=len(id2label)), model,
                                           SimpleLossCompute(criterion,  optimizer,scheduler), train=False, id2label=id2label)
        print('Loss:', loss)
        testResult.append(f)
    valBest=max(valResult)
    print('ValBest epoch:', [i for i, j in enumerate(valResult) if j == valBest])
    testBest = max(testResult)
    print('TestBest epoch:', [i for i, j in enumerate(testResult) if j == testBest])


trainSents=preProcess(readData('dps/swbd/train'))
valSents=preProcess(readData('dps/swbd/val'))
testSents=preProcess(readData('dps/swbd/test'))
label2id,id2label=build_vocab(trainSents)
print(id2label)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
trainData=idData(tokenizer,trainSents,label2id)
valData=idData(tokenizer,valSents,label2id)
testData=idData(tokenizer,testSents,label2id)
encoder=Encoder(INPUT_DIM,HIDDEN_DIM,ENCODER_LAYER,len(id2label),DROUPOUT_RATE).to(device)

run(EPOCH,encoder,BATCH_SIZE,trainData,valData,testData,id2label,tokenizer._convert_token_to_id('[PAD]'))