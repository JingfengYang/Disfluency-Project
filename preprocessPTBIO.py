import os

REPARANDUM=0
REPAIR=1

def isRepair(inDisf):
    ret=True
    for t in inDisf:
        if t[1]==REPARANDUM:
            ret=False
            break
    return ret

def readData(dir_name):
    sentList=[]

    for file in os.listdir(dir_name):
        with open(dir_name + '/' + file, 'r') as reader:
            start = 0
            for line in reader:

                if start==0 and line.strip().startswith('===='):
                    start=1
                    continue
                if start==0:
                    continue
                if line.strip()=='':
                    continue
                if line.strip().startswith('SpeakerA') or line.strip().startswith('SpeakerB'):
                    continue
                tokens=line.strip().split()
                if not tokens[-1]=='E_S' and not tokens[-1]=='N_S':
                        #and not line.strip().startswith('SpeakerA') and not line.strip().startswith('SpeakerB'):
                    #if not tokens[0]=='MUMBLEx/XX':
                        #print(line)
                    continue
                sent1 = []
                sent2 = []
                if tokens[-1] == 'E_S' or tokens[-1] == 'N_S':
                    tokens=tokens[:-1]
                inDisf=[]
                iLayer=0
                for token in tokens:
                    if token=='[':

                        inDisf.append([iLayer,REPARANDUM])
                        iLayer+=1
                    elif token==']':
                        inDisf.pop()
                        iLayer-=1
                    elif token=='+':
                        if len(inDisf)==0:
                            print(line)
                        inDisf[-1][1]=REPAIR
                    elif len(inDisf)==0:
                        sent1.append([token,'None'])
                    else:
                        if isRepair(inDisf):
                            sent1.append([token,'-'])
                        else:
                            sent1.append([token, '+'])
                assert(len(inDisf)==0 and iLayer==0)
                isBracket=0
                for token in sent1:
                    if token[0][0]=='{':

                        assert(len(token)==2)
                        isBracket=1
                        continue
                    elif token[0]=='}':
                        isBracket=0
                    else:
                        assert(not token[0].startswith('{') and not token[0].startswith('}'))
                        assert(not token[0].startswith('/'))
                        t=token[0].split('/')

                        assert(len(t)==2)
                        sent2.append([t[0].lower(),t[1],token[1]])
                assert(isBracket==0)
                if len(sent2)>0:
                    sentList.append(sent2)

    return sentList

def preProcess(sents):
    newSents=[]
    for sent in sents:
        sent1=[]
        sent2=[]
        for t in sent:
            if not t[0].endswith('-') and not t[1]=='.' and not t[1]==',':
                sent1.append(t)
        if len(sent1)==0:
            continue
        #tag = 0
        for (i,t) in enumerate(sent1):
            if t[2]=='+':
                #tag = 1
                if i==0 or not sent1[i-1][2]=='+':
                    if i==len(sent1)-1 or not sent1[i+1][2]=='+':
                        sent2.append([t[0], t[1], 'I'])
                    else:
                        sent2.append([t[0],t[1],'I'])
                elif i==len(sent1)-1 or not sent1[i+1][2]=='+':
                    sent2.append([t[0], t[1], 'I'])
                else:
                    sent2.append([t[0], t[1], 'I'])
            elif t[2]=='-':
                sent2.append([t[0], t[1], 'O'])
            else:
                assert(t[2]=='None')
                sent2.append([t[0], t[1], 'O'])
        newSents.append(sent2)
        #if tag==1:
            #newSents.append(sent2)
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

def writeTransSents(sents,src_file,trg_file,lab_file):
    with open(src_file,'w') as src_writer, open(trg_file,'w') as trg_writer, open(lab_file,'w') as lab_writer:
        for sent in sents:
            src_sent=[]
            trg_sent=[]
            lab_sent=[]
            for t in sent:
                src_sent.append(t[0])
                trg_sent.append(t[0] if t[2]=='OR' or t[2]=='O' else 'EI')
                lab_sent.append('O' if t[2]=='OR' or t[2]=='O' else 'EI')
            src_writer.write(' '.join(src_sent)+'\n')
            trg_writer.write(' '.join(trg_sent)+'\n')
            lab_writer.write(' '.join(lab_sent)+'\n')

if __name__ == "__main__":
    t = readData('dps/swbd/train')
    print(len(t))
    trainSents = preProcess(t)
    valSents = preProcess(readData('dps/swbd/val'))
    testSents = preProcess(readData('dps/swbd/test'))
    writeTransSents(trainSents, 'src-train.txt','tgt-train.txt','lab-train.txt')
    writeTransSents(valSents, 'src-val.txt','tgt-val.txt','lab-val.txt')
    writeTransSents(testSents, 'src-test.txt','tgt-test.txt','lab-test.txt')
    ##print(stat(trainSents), len(trainSents))
