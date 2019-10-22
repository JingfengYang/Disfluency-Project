import os
import torch


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def readData(dir_name):
    sentList = []
    sent = []
    label = set()
    for file in os.listdir(dir_name):
        with open(dir_name+'/'+file, 'r') as reader:
            for line in reader:
                content = line.strip()
                if content == '':
                    sentList.append(sent)
                    sent = []
                else:
                    tokens=content.split()
                    label.add(tokens[6])
                    sent.append([tokens[2],tokens[3], tokens[6]])
        if len(sent) > 0:
            sentList.append(sent)
            sent = []

    return sentList

def preProcess(sents):
    newSents=[]
    for sent in sents:
        sent1=[]
        sent2=[]
        for t in sent:
            if not t[0].endswith('-') and not t[0]=='SILENCE' and not t[0]=='TRACE' and not t[1]=='None' and not t[0]=='uh' and not t[0]=='um':
                sent1.append(t)
        if len(sent1)<=0:
            continue
        tag = 0
        for (i,t) in enumerate(sent1):
            if t[2]=='+':
                tag = 1
                if i==0 or not sent1[i-1][2]=='+':
                    if i==len(sent1)-1 or not sent1[i+1][2]=='+':
                        sent2.append([t[0], t[1], 'BE_IP'])
                    else:
                        sent2.append([t[0],t[1],'BE'])
                elif i==len(sent1)-1 or not sent1[i+1][2]=='+':
                    sent2.append([t[0], t[1], 'IP'])
                else:
                    sent2.append([t[0], t[1], 'IE'])
            elif t[2]=='-':
                sent2.append([t[0], t[1], 'OR'])
            else:
                assert(t[2]=='None')
                sent2.append([t[0], t[1], 'O'])
        if tag==1:
            newSents.append(sent2)
    return newSents

def stat(sents):
    difsents=0
    muti=0
    nomo=0
    for sent in sents:
        tag=0
        for t in sent:
            if t[2]=='BE':
                muti+=1
                tag=1
            if t[2]=='BE_IP':
                nomo+=1
                tag=1
        if tag==1:
            difsents+=1
    return muti,nomo,muti+nomo,difsents



def writeSents(sents,file):
    with open(file,'w') as writer:
        for sent in sents:
            for seq in list(zip(*sent)):
                writer.write('\t'.join(seq)+'\n')
            writer.write('\n')

def build_vocab(sents,min_count=0):###
    labels=set()

    for sent in sents:
        for token in sent:
            labels.add(token[2])
    id2label=list(labels)
    #id2label=id2label+["<pad>"]
    label2id={}
    for i,label in enumerate(id2label):
        label2id[label]=i

    return label2id,id2label

def get_idx(word,vocDic,unk=1):
    if word in vocDic:
        return vocDic[word]
    else:
        return len(vocDic)

def idData(tokenizer,sents,tagDic):
    #print(tagDic)
    sents=sorted(sents,key=lambda x:len(x),reverse=True)
    allHeads=[]
    allSents=[]
    allTags=[]
    allTagMask=[]
    for sent in sents:
        assert(len(sent)>0)
        '''if len(sent)<=0:
            print(sent)
            continue'''
        words=[token[0] for token in sent]
        tags=[token[2] for token in sent]
        words=["[CLS]"] + words + ["[SEP]"]
        tags=["<pad>"] + tags + ["<pad>"]
        idSent=[]
        idHead=[]
        idTag=[]
        idTagMaxsk=[]
        for w,t in zip(words, tags):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)

            if w in ("[CLS]", "[SEP]"):
                is_head=[0]
            else:
                is_head = [1] + [0] * (len(tokens) - 1)

            ts=[]
            if not w in ("[CLS]", "[SEP]"):
                ts = [t]
            tmask=[]
            if not w in ("[CLS]", "[SEP]"):
                tmask = [1]
            #print(t)
            yy = [get_idx(each,tagDic) for each in ts]  # (T,)

            idSent.extend(xx)
            idHead.extend(is_head)
            idTag.extend(yy)
            idTagMaxsk.extend(tmask)
        allSents.append(idSent)
        allTags.append(idTag)
        allHeads.append(idHead)
        allTagMask.append(idTagMaxsk)
        #allSents.append([) for token in sent])
        #allTags.append([tagDic[token[2]] for token in sent])
    #print(allTags)
    return [allSents,allTags,allHeads,allTagMask]

def padding(batch,pad=0):
    lengths=[len(sent) for sent in batch]
    ntokens=sum(lengths)
    maxLen=max(lengths)
    outbatch=[]
    for sent in batch:
        outbatch.append(sent+[pad]*(maxLen-len(sent)))

    return [torch.tensor(outbatch,dtype=torch.long, device=device),torch.tensor(lengths,dtype=torch.long, device=device),ntokens]

def batchIter(data,batch_size,w_tag_pad=0,t_tag_pad=0,train=True):
    allSents=data[0]
    allTags=data[1]
    allHead=data[2]
    allTagMask=data[3]
    nbatch=len(allSents)//batch_size
    for s,t,h,m in zip(allSents,allTags,allHead,allTagMask):
        assert(len(s)==len(h))
        assert(len(t)==len(m))
    for i in range(nbatch):
        yield padding(allSents[i*batch_size:(i+1)*batch_size],pad=w_tag_pad), padding(allTags[i*batch_size:(i+1)*batch_size],pad=t_tag_pad), padding(allHead[i*batch_size:(i+1)*batch_size],pad=0)\
            , padding(allTagMask[i * batch_size:(i + 1) * batch_size], pad=0)
    if len(allSents)>nbatch*batch_size:
        yield padding(allSents[nbatch * batch_size:],pad=w_tag_pad), padding(allTags[nbatch * batch_size:],pad=t_tag_pad), padding(allHead[nbatch * batch_size:],pad=0) \
            , padding(allTagMask[nbatch * batch_size:], pad=0)

def isEdit(id,id2label):
    if not id2label[id]=='O' and not id2label[id]=='OR':
        return True
    else:
        return False
'''trainSents=preProcess(readData('train'))
valSents=preProcess(readData('val'))
testSents=preProcess(readData('test'))
writeSents(trainSents,'train.txt')
writeSents(valSents,'val.txt')
writeSents(testSents,'test.txt')
print(stat(trainSents),len(trainSents))'''