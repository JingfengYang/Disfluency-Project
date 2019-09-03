'''
PART ONE:
Terminals Parsing
'''
import os
import xml.etree.ElementTree as ET
import sys


#get_IDdict will return built up IDdict and IDlist
def get_IDdict(root, IDdict, IDlist):
    for child in root:
        if child.tag == 'word':
            wordid = child.get(namespaceIdentifier+'id')
            IDdict[wordid] = []
            IDlist.append(wordid)
            #attach the word
            IDdict[wordid].append(child.get('orth'))
            #attach pos tag
            IDdict[wordid].append(child.get('pos'))
            #attach start end time
            IDdict[wordid].append(child.get(namespaceIdentifier+'start'))
            IDdict[wordid].append(child.get(namespaceIdentifier+'end'))

            #build Phonedict if link exists
            phoneword = child.find(namespaceIdentifier+'pointer')
            if phoneword != None:
                phoneword_ID = phoneword.get('href').split('#')[1][3:-1]
                Phoneword_dict[phoneword_ID] = wordid
            else:
                continue

        if child.tag == 'punc':
            wordid = child.get(namespaceIdentifier+'id')
            IDdict[wordid] = []
            #attach the word
            IDlist.append(wordid)
            IDdict[wordid].append(child.text)
            #attach pos tag
            IDdict[wordid].append(None)
            #attach start end time
            IDdict[wordid].append(None)
            IDdict[wordid].append(None)
        if child.tag == 'sil':
            wordid = child.get(namespaceIdentifier+'id')
            IDdict[wordid] = []
            IDlist.append(wordid)
            IDdict[wordid].append('SILENCE')
            #attach pos tag
            IDdict[wordid].append(None)
            #attach start end time
            IDdict[wordid].append(None)
            IDdict[wordid].append(None)
        if child.tag == 'trace':
            wordid = child.get(namespaceIdentifier+'id')
            IDdict[wordid] = []
            IDlist.append(wordid)
            IDdict[wordid].append('TRACE')
            #attach pos tag
            IDdict[wordid].append(None)
            #attach start end time
            IDdict[wordid].append(None)
            IDdict[wordid].append(None)
        else:
            continue
    return IDdict, IDlist

# print out sentence with word-level attributes
# print out with space between sentences
def pretty_print(AIDdict, AIDlist, BIDdict, BIDlist):
    indexA = 0
    indexB = 0
    inwhich = ''
    if AIDlist[0][1:].split('_')[0] == '1':
        inwhich = 'A'
    else:
        inwhich = 'B'

    while indexA < len(AIDlist) - 1 or indexB < len(BIDlist) - 1:
        if inwhich == 'A':
            if indexA >= len(AIDlist) - 1 and indexB < len(BIDlist):
                print 'A', AIDlist[indexA],
                for element in AIDdict[AIDlist[indexA]]:
                    if type(element) is tuple:
                        for subele in element:
                            print subele,
                    elif type(element) is list:
                        for subele in element:
                            print subele,
                    else:
                        print element,
                print ""
                inwhich = 'B'
                print ''
                continue

            print 'A', AIDlist[indexA],
            for element in AIDdict[AIDlist[indexA]]:
                if type(element) is tuple:
                    for subele in element:
                        print subele,
                elif type(element) is list:
                    for subele in element:
                        print subele,
                else:
                    print element,
            print ""
            nextsentnum = int(AIDlist[indexA + 1].split('_')[0][1:])
            sentnum = int(AIDlist[indexA].split('_')[0][1:])
            if nextsentnum - sentnum > 1:
                inwhich = 'B'
                print ''
            if nextsentnum - sentnum == 1:
                print ''
            indexA += 1
            # if indexA >= len(AIDlist) and indexB >= len(BIDlist):
            #     break

        if inwhich == 'B':
            if indexB >= len(BIDlist) - 1 and indexA < len(AIDlist):
                print 'B', BIDlist[indexB],
                for element in BIDdict[BIDlist[indexB]]:
                    if type(element) is tuple:
                        for subele in element:
                            print subele,
                    elif type(element) is list:
                        for subele in element:
                            print subele,
                    else:
                        print element,
                print ""
                inwhich = 'A'
                print ''
                continue

            print 'B', BIDlist[indexB],
            for element in BIDdict[BIDlist[indexB]]:
                if type(element) is tuple:
                    for subele in element:
                        print subele,
                elif type(element) is list:
                    for subele in element:
                        print subele,
                else:
                    print element,
            print ""
            nextsentnum = int(BIDlist[indexB + 1].split('_')[0][1:])
            sentnum = int(BIDlist[indexB].split('_')[0][1:])
            if nextsentnum - sentnum > 1:
                inwhich = 'A'
                print ''
            if nextsentnum - sentnum == 1:
                print ''
            indexB += 1
            # if indexA >= len(AIDlist) and indexB >= len(BIDlist):
            #     break


def attach_to_terminal_func(termi_attribute_dict, IDdict):
    for ID in IDdict:
        IDdict[ID].append(termi_attribute_dict[ID])

def None_dialfile_dict_builder(IDdict):
    termi_dialAct_dict = {}
    #second attach termi_wordID who has not been attached in diaActdict
    for key in IDdict:
        if key not in termi_dialAct_dict:
            termi_dialAct_dict[key] = (None, None, None)
    return termi_dialAct_dict

def get_dialActDict(root):
    #diaActdict structure:
    #{(daID, niteType, swbdType): [word_id, word_id, ..., word_id]}
    diaActdict = {}
    for child in root:
        #find dialAct first
        niteType = child.get('niteType')
        swbdType = child.get('swbdType')
        daId = child.get(namespaceIdentifier+'id')
        diaActdict[(daId, niteType, swbdType)] = []
        pointers_to_word = child.findall(namespaceIdentifier+'child')
        for pointer in pointers_to_word:
            word_id = pointer.get('href').split('#')[1][3:-1]
            diaActdict[(daId, niteType, swbdType)].append(word_id)
    return diaActdict

#attach diaAct to word list
def attach_diaAct_to_terminal(termi_dial_dict, IDdict):
    for key in IDdict:
        IDdict[key].append(termi_dial_dict[key])



def terminal_diaAct_dict_builder(diaActdict, IDdict):
    #termi_dialAct_dict structure:
    #{terminal_wordID: (daID, niteType, swbdType)}
    termi_dialAct_dict = {}

    #first attach diaActdict termi_wordID
    for key in diaActdict:
        for word_id in diaActdict[key]:
            termi_dialAct_dict[word_id] = key

    #second attach termi_wordID who has not been attached in diaActdict
    for key in IDdict:
        if key not in termi_dialAct_dict:
            termi_dialAct_dict[key] = (None, None, None)

    return termi_dialAct_dict

def None_dflfile_dict_builder(IDdict):
    termi_dfl_dict = {}
    for key in IDdict:
        if key not in reparandum_dict and key not in repair_dict:
            termi_dfl_dict[key] = None

    return termi_dfl_dict

def proDisf(disf,reparandum_dict):
    for child in disf:
        if child.tag=='repair':
            for c in child:
                if c.tag==(namespaceIdentifier + 'child'):
                    reparandum_dict[c.get('href').split('#')[1][3:-1]]='-'
                else:
                    assert(c.tag=='disfluency')
                    proDisf(c,reparandum_dict)
        else:
            assert(child.tag=='reparandum')


def get_dfl_dict(root):
    reparandum_dict = {}
    repair_dict = {}
    for child in root:
        #since disfluency is in tree structrue, the depth are not decided
        #we use iter() to convert every disfluency child into a list.
        all_children = list(child.iter())
        reparandum_depth = 1


        for subchild in all_children:
            if subchild.tag == 'reparandum' or 'repair':
                words = []
                termis = subchild.findall(namespaceIdentifier + 'child')
                for word in termis:
                    words.append(word.get('href').split('#')[1][3:-1])

                for i in range(len(words)):
                    reparandum_dict[words[i]] = '+'
    for child in root:
        assert(child.tag=='disfluency')
        proDisf(child,reparandum_dict)

    return reparandum_dict, repair_dict


#create terminals disfluency dict
def terminal_dfl_dict_builder(reparandum_dict, repair_dict, IDdict):
    #termi_dfl_dict structure:
    #{termi_wordID: disfluency_label}
    termi_dfl_dict = {}
    for key in reparandum_dict:
        termi_dfl_dict[key] = reparandum_dict[key]
    for key in repair_dict:
        termi_dfl_dict[key] = repair_dict[key]

    for key in IDdict:
        if key not in reparandum_dict and key not in repair_dict:
            termi_dfl_dict[key] = None

    return termi_dfl_dict

'''======================part_one======================'''
#namespace is retrieved by hand ahead, it's correct
namespaceIdentifier = '{http://nite.sourceforge.net/}'

#for iteration purpose, we split filename according to
#their name pattern, only the first part varies
swnumb = sys.argv[1]

#use ET package retrieve tree structure data for A and B speaker
Afilepath = os.path.join(os.getcwd(), 'terminals', swnumb + '.A.terminals.xml')
Bfilepath = os.path.join(os.getcwd(), 'terminals', swnumb + '.B.terminals.xml')
Atree = ET.parse(Afilepath)
Btree = ET.parse(Bfilepath)

Aroot = Atree.getroot()
Broot = Btree.getroot()

#IDdict is a dictionary for quick checking attribute of each word
#IDdict structure:
#{terminal_wordID: ['word', 'pos', 'starttime', 'endtime', ]}
AIDdict = {}
BIDdict = {}

#IDlist is an array, for sequence record, because IDdict will loss sequence info
AIDlist = []
BIDlist = []

#phoneword_dict is a dict to link between terminal and phonewords transcripts
#we don't distinguish A and B for A and B has different wordID, they won't conflict
Phoneword_dict = {}

AIDdict, AIDlist = get_IDdict(Aroot, AIDdict, AIDlist)
BIDdict, BIDlist = get_IDdict(Broot, BIDdict, BIDlist)
'''======================part_two======================'''

try:
    #adding dialogue act tags into original dataset

    Afilepath = os.path.join(os.getcwd(), 'dialAct',swnumb + '.A.dialAct.xml')
    Bfilepath = os.path.join(os.getcwd(), 'dialAct',swnumb + '.B.dialAct.xml')

    Atree = ET.parse(Afilepath)
    Aroot = Atree.getroot()
    Btree = ET.parse(Bfilepath)
    Broot = Btree.getroot()
    #get dialAct dictionary for speaker A and B
    A_dial_Act_dict = get_dialActDict(Aroot)
    B_dial_Act_dict = get_dialActDict(Broot)

    #get termi_diaAct_dict
    Atermi_dialAct_dict = terminal_diaAct_dict_builder(A_dial_Act_dict, AIDdict)
    Btermi_dialAct_dict = terminal_diaAct_dict_builder(B_dial_Act_dict, BIDdict)

    # #attach to terimal_wordID for pretty print
    # attach_diaAct_to_terminal(Atermi_dialAct_dict, AIDdict)
    # attach_diaAct_to_terminal(Btermi_dialAct_dict, BIDdict)


    # pretty_print(AIDdict, AIDlist, BIDdict, BIDlist)
except:
    Atermi_dialAct_dict = None_dialfile_dict_builder(AIDdict)
    Btermi_dialAct_dict = None_dialfile_dict_builder(BIDdict)

'''======================part_three======================'''
Afilepath = os.path.join(os.getcwd(), 'disfluency', swnumb+'.A.disfluency.xml')
Bfilepath = os.path.join(os.getcwd(), 'disfluency', swnumb+'.B.disfluency.xml')
Atree = ET.parse(Afilepath)
Aroot = Atree.getroot()
Btree = ET.parse(Bfilepath)
Broot = Btree.getroot()


    #create 2 list to record the position of reparandum and repair in
    #terminal

    #get reparandum_dict and repair_dict
Areparandum_dict, Arepair_dict = get_dfl_dict(Aroot)
Breparandum_dict, Brepair_dict = get_dfl_dict(Broot)

    #link termi_wordID to reparandum and repair
Atermi_dfl_dict = terminal_dfl_dict_builder(Areparandum_dict, Arepair_dict, AIDdict)
Btermi_dfl_dict = terminal_dfl_dict_builder(Breparandum_dict, Brepair_dict, BIDdict)

    # #attach reparandum/repair for pretty print
    # attach_to_terminal_func(Atermi_dfl_dict, AIDdict)
    # attach_to_terminal_func(Btermi_dfl_dict, BIDdict)

    # pretty_print(AIDdict, AIDlist, BIDdict, BIDlist)







'''======================combination======================'''

attach_to_terminal_func(Atermi_dfl_dict, AIDdict)
attach_to_terminal_func(Btermi_dfl_dict, BIDdict)

pretty_print(AIDdict, AIDlist, BIDdict, BIDlist)

