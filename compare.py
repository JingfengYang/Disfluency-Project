predTags=[]
with open('predictTest') as reader:
    for line in reader:
        a = ['O' if x == 'OR' else x for x in line.strip().split()]
        predTags.append(a)

goldTags=[]
words=[]
with open('test.txt') as reader:
    i=0
    for line in reader:
        if i%4==0:

            words.append(line.strip().split())
        if i%4==2:
            a=['O' if x == 'OR' else x for x in line.strip().split()]
            goldTags.append(a)
        i+=1
assert(len(predTags)==len(goldTags))
count=0

for pred,gold,word in zip(predTags,goldTags,words):
    assert(len(pred)==len(gold))
    if not pred==gold:
        count+=1
        for p,g,w in zip(pred,gold,word):
            print(w,g,p)
        print('\n')
print(count)