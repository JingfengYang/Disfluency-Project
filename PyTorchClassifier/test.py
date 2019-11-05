import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertConfig, WarmupLinearSchedule
from transformers import AdamW, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import os
import numpy as np
import matplotlib.pyplot as plt


def tokenize_id_pad_mask_data(tokenizer, file, tagDic, max_len=128):
    # Returns list of tokenized, id converted lists- allSents
    # and list of all tags- allTags
    allSents = []
    allTags = []
    print("Reading Data")
    with open(file) as f:
        content = [line.strip() for line in f.readlines()]
    lines = []
    sentence_with_tag = []
    for line in content:
        lines.append(line)
        if line == "":
            if len(lines) != 4:
                lines = []
                continue
            # Line 1 is words
            words = lines[0].split()
            if len(words) <= 3:
                lines = []
                continue
            # Line 2 is POS tags Throw POS tags
            # Line 3 is disfluent tags
            tags = lines[2].split()
            if tags.count('O') + tags.count('OR') == len(tags):
                tag = 'FLUENT'
            else:
                tag = 'DISFLUENT'
            sentence_with_tag.append([words, tag])
            lines = []
    # sentence_with_tag.sort(key=lambda x: len(x[0]), reverse=True)
    rawSents = [" ".join(sent[0]) for sent in sentence_with_tag]
    print("Collected {} valid sentences".format(len(rawSents)))
    print("Tokenizing and Converting to ID")
    for words, tag in sentence_with_tag:
        words = ["[CLS]"] + words + ["[SEP]"]
        idSent = []
        for w in words:
            # Tokenize and convert to id
            tokens = tokenizer.tokenize(w) if w not in (
                "[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)
            idSent.extend(xx)
        allSents.append(idSent)
        allTags.append(tagDic[tag])

    print("Adding Padding")
    padSents = []
    masks = []
    padToken = tokenizer._convert_token_to_id('[PAD]')
    for sent in allSents:
        if len(sent) > max_len:
            padSents.append(sent[:max_len])
            masks.append([1] * max_len)
        else:
            padSents.append(sent + [padToken] * (max_len - len(sent)))
            masks.append([1] * len(sent) + [0] * (max_len - len(sent)))
    return [padSents, masks, allTags], rawSents


def getDataLoader(data, sampler="random", batch_size=2):
    inputs, masks, labels = torch.tensor(data[0]), \
        torch.tensor(data[1]), torch.tensor(data[2])
    data = TensorDataset(inputs, masks, labels)
    if sampler == "sequential":
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)
    dataloader = DataLoader(
        data, sampler=sampler, batch_size=batch_size)
    return dataloader


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def test(dataloader, model):
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    all_labels = []

    # Evaluate data for one epoch
    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        # Move logits and labels to CPU
        logits = logits[0].detach().cpu().numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        all_labels.extend(list(pred_flat))

    return all_labels


def evaluate(dataloader, model):
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Tracking variables
    eval_accuracy, nb_eval_steps = 0, 0

    # Evaluate data for one epoch
    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        # Move logits and labels to CPU
        logits = logits[0].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("Accuracy: {}".format(eval_accuracy / nb_eval_steps))


if __name__ == '__main__':
    # Specify the CPU/GPU as the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating label vocabulary.")
    id2label = ['DISFLUENT', 'FLUENT']
    label2id = {}
    for i, label in enumerate(id2label):
        label2id[label] = i

    print("Loading BERT Tokenizer")
    # tokenizer = BertTokenizer.from_pretrained(
    #     'bert-base-uncased', do_lower_case=True)
    tokenizer = BertTokenizer.from_pretrained('./tokenizer/')

    print("Loading BERT model.")
    # model = BertForSequenceClassification.from_pretrained(
    #     "bert-base-uncased", num_labels=2)
    model = BertForSequenceClassification.from_pretrained('./model/')
    model.cuda()

    print("Loading Test Data")
    testData, rawSents = tokenize_id_pad_mask_data(
        tokenizer, 'skysports_extracted.txt', label2id)
    testDataLoader = getDataLoader(testData, sampler="sequential")
    labels = test(testDataLoader, model)

    # Evaluate on Val and Test
    # print("Loading Validation Data")
    # valData, _ = tokenize_id_pad_mask_data(tokenizer, 'val.txt', label2id)
    # trainData = valData
    # print("Loading Test Data")
    # testData, _ = tokenize_id_pad_mask_data(tokenizer, 'test.txt', label2id)
    # valDataLoader = getDataLoader(valData)
    # testDataLoader = getDataLoader(testData)
    # evaluate(valDataLoader, model)
    # evaluate(testDataLoader, model)

    disfluent = []
    with open('predictions.txt', 'w+') as f:
        for tag, sent in zip(labels, rawSents):
            f.write(sent + '\n')
            f.write(id2label[tag] + '\n')
            f.write('\n')
            if tag == 0:
                disfluent.append(sent)

    with open('disfluent.txt', 'w+') as f:
        for i in range(len(disfluent)):
            f.write("{}\n".format(disfluent[i]))
