import random
from collections import Counter
from nltk.tokenize import word_tokenize

sents=[]
tags=[]
counts=[1,2,3]
actions=[0,1]
wordCounts=[1,2,3,4,5,6]
punc_list=['.','?','!',';',':','-','--','(',')','[',']','{','}',"'",'"','...',',']

def insert(pos, count,sent,ngrams,tagSent):
    sent[pos:pos]=random.choice(ngrams[count]).split()
    tagSent[pos:pos]=['I']*count

def repeat(pos,count,sent,tagSent):
    sent[pos:pos]=sent[pos:pos+count]
    tagSent[pos:pos] = ['I'] * count

stats=[]
for i in range(7):
    stats.append(Counter())

id=0
with open('news.2016.en.shuffled') as reader:
    for line in reader:
        if id%10000==0:
            print(id)
        id+=1
        sent=line.strip()
        sent=word_tokenize(sent)
        sent=[token for token in sent if not token in punc_list]
        if len(sent)==0:
            continue
        for i in wordCounts:
            if i==1:
                stats[i].update(sent)
            else:
                if len(sent)>=i:
                    for j in range(len(sent)-i+1):
                        stats[i][' '.join(sent[j:j+i])]+=1
        if len(sent)>0:
            sents+=[sent]
            tags+=[['O']*len(sent)]
print ('finish read')
ngrams=[[]]
for i in wordCounts:
    ngrams.append([gram for gram,_ in stats[i].most_common(10000)])

print('finish count')

for sent,tagSent in zip(sents,tags):
    count=random.choice(counts)
    allPos = range(len(sent))
    poses=random.choices(allPos,k=count)
    poses.sort(reverse = True)
    for pos in poses:
        action = random.choice(actions)
        wordCount=random.choice(wordCounts)
        if action==1:
            insert(pos,wordCount,sent,ngrams,tagSent)
        else:
            repeat(pos, wordCount, sent,tagSent)

with open('fakeData_woPunc.txt','w') as writer:
    for sent,tagSent in zip(sents,tags):
        writer.write(' '.join(sent)+'\n')
        writer.write(' '.join(['P']*len(sent)) + '\n')
        writer.write(' '.join(tagSent) + '\n')
        writer.write('\n')
