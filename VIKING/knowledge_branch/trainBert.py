import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer)
from transformers import AdamW, WarmupLinearSchedule

trainDataPath = './Cache/qaFromComments_train.csv'
valDataPath = './Cache/qaFromComments_val.csv'
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
maxLen = 512

pretrained_weights = 'bert-base-uncased'    
config = BertConfig.from_pretrained(pretrained_weights,
                                          num_labels=2)
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
model = BertForSequenceClassification.from_pretrained(pretrained_weights,config=config)

class QAClassifierDataset(Dataset):
    def __init__(self, Q, T, L):
        X = [' '.join(['[CLS]', Q[i], '[SEP]', T[i], '[SEP]']) for i in range(len(T))]
        self.L = [int(l) for l in L]
        
        self.input_ids = []
        self.attention_mask = []
        self.token_type_ids = []

        for x in tqdm(X):
            tokenized_text = tokenizer.tokenize(x)
            # cut long paragraphs
            if len(tokenized_text) > maxLen:
                tokenized_text = tokenized_text[:maxLen]
                tokenized_text[maxLen-1] = '[SEP]'
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            eos1 = 0
            eos2 = 0
            for i in range(len(tokenized_text)):
                char = tokenized_text[i]
                if char == '[SEP]':
                    if eos1 == 0:
                        eos1 = i
                    else:
                        eos2 = i
                        break
#             print(eos1, eos2)
            indexed_tokens = indexed_tokens + [0] * (maxLen - len(indexed_tokens))
            self.input_ids.append(indexed_tokens)
            att = np.ones(len(indexed_tokens))
            att[eos2+1:] = 0
            types = np.ones(len(indexed_tokens))
            types[:eos1+1] = 0

            self.attention_mask.append(att)
            self.token_type_ids.append(types)
        print('loaded '+str(len(self.input_ids))+' lines.')
    def __len__(self):
        return len(self.L)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx], self.L[idx]


train_batch_size = 8
eval_batch_size = 8
adam_epsilon = 1e-8
learning_rate = 5e-5 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
def collate_fn(batch):
    input_ids, attention_mask, token_type_ids, L = zip(*batch)
    return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(token_type_ids), torch.LongTensor(L)

def train(train_dataset, model):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, sampler=train_sampler, batch_size=train_batch_size)

    t_total = len(train_dataloader) // 1 * 3

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=t_total)
    # Train!
    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Num Epochs = %d", 5)
    print("  Instantaneous batch size per GPU = %d", 8)
    print("  Gradient Accumulation steps = %d", 1)
    print("  Total optimization steps = %d", t_total)

    model.zero_grad()
    
    for e in range(5):
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        for batch in tqdm(train_dataloader):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            loss.backward()

            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            print(loss.item())
        print(e, global_step, tr_loss / global_step)
    return global_step, tr_loss / global_step

def eval(eval_dataset, model):      

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, sampler=eval_sampler, batch_size=eval_batch_size)  
    # Eval!
    print("***** Running evaluation *****")
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    correct = 0
    total = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        batch_res = logits.detach().cpu().numpy()
        batch_res = np.argmax(batch_res, axis=1)
        labels = batch[3].detach().cpu().numpy()
        correct += np.sum(labels == batch_res)
        total += len(labels)
        print(correct, total)
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    return preds

#train
Q, T, L = [], [], []
with open(trainDataPath) as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            Q.append(row[0])
            T.append(row[1])
            L.append(row[2])
            line_count += 1
    print(f'Processed {line_count} lines.')
train_dataset = QAClassifierDataset(Q, T, L)
train(train_dataset, model)
Q, T, L = [], [], []

cache_dir = "Models/Bert/"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
model.save_pretrained(cache_dir)
# #val
# with open(valDataPath) as csvfile:
#     csv_reader = csv.reader(csvfile, delimiter=',')
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             line_count += 1
#         else:
#             Q.append(row[0])
#             T.append(row[1])
#             L.append(row[2])
#             line_count += 1
#     print(f'Processed {line_count} lines.')
# val_dataset = QAClassifierDataset(Q, T, L)
# eval(val_dataset, model)