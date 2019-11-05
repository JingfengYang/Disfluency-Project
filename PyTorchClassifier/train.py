import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt


def tokenize_id_pad_mask_data(tokenizer, file, tagDic, max_len=128):
    # Returns list of tokenized, id converted lists- allSents
    # and list of all tags- allTags
    allSents = []
    allTags = []
    with open(file) as f:
        content = [line.strip() for line in f.readlines()]
    lines = []
    sentence_with_tag = []
    for line in content:
        lines.append(line)
        if line == "":
            # Line 1 is words
            words = lines[0].split()
            # Line 2 is POS tags Throw POS tags
            # Line 3 is disfluent tags
            tags = lines[2].split()
            if tags.count('O') + tags.count('OR') == len(tags):
                tag = 'FLUENT'
            else:
                tag = 'DISFLUENT'
            sentence_with_tag.append([words, tag])
            lines = []
    sentence_with_tag.sort(key=lambda x: len(x[0]), reverse=True)
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

    # Add Padding
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
    return [padSents, masks, allTags]


def getDataLoader(data, batch_size=32):
    inputs, masks, labels = torch.tensor(data[0]), \
        torch.tensor(data[1]), torch.tensor(data[2])
    data = TensorDataset(inputs, masks, labels)
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
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("Accuracy: {}".format(eval_accuracy / nb_eval_steps))


def runner(train_dataloader, validation_dataloader, model, optimizer,
           device, epochs=4):
    # Store our loss and accuracy for plotting
    train_loss_set = []

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):

        # Training
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss = model(b_input_ids, token_type_ids=None,
                         attention_mask=b_input_mask, labels=b_labels)
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # Validation
        test(validation_dataloader, model)


if __name__ == '__main__':
    # Specify the CPU/GPU as the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating label vocabulary.")
    id2label = ['DISFLUENT', 'FLUENT']
    label2id = {}
    for i, label in enumerate(id2label):
        label2id[label] = i

    print("Loading BERT Tokenizer")
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case=True)
    print("Loading Train Data")
    trainData = tokenize_id_pad_mask_data(tokenizer, 'train.txt', label2id)
    print("Loading Validation Data")
    valData = tokenize_id_pad_mask_data(tokenizer, 'val.txt', label2id)
    print("Loading Test Data")
    testData = tokenize_id_pad_mask_data(tokenizer, 'test.txt', label2id)

    trainDataLoader = getDataLoader(trainData)
    valDataLoader = getDataLoader(valData)
    testDataLoader = getDataLoader(testData)

    print("Loading BERT model.")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2)
    model.cuda()

    print("Creating optimizer with model parameters.")
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
    ]
    # This variable contains all of the hyperparemeter
    # information our training loop needs
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=2e-5,
                      warmup=.1)

    print("Starting Training")
    # Number of training epochs (authors recommend between 2 and 4)
    runner(trainDataLoader, valDataLoader, model, optimizer,
           device, epochs=4)


# Get Sentences List and Tags
# sentences = ["Did that John showed up please you?",
#              "It is important for the more you eat, the more."]
# labels = [0, 0]

# sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]


# tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
# print ("Tokenize the first sentence:")
# print (tokenized_texts[0])


# optimizer = BertAdam(optimizer_grouped_parameters,
#                      lr=2e-5,
#                      warmup=.1)

# # Function to calculate the accuracy of our predictions vs labels
