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

testDataPath = './Cache/bert_predicted_kg.csv'
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
maxLen = 512

pretrained_weights = 'bert-base-uncased'    
config = BertConfig.from_pretrained('./Models/Bert', num_labels=2)
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
model = BertForSequenceClassification.from_pretrained('./Models/Bert',config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

Q, T, A, L = [], [], [], []
with open(testDataPath) as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            Q.append(row[0])
            T.append(row[1])
            A.append(row[2])
            L.append(row[3])
            line_count += 1
    print(f'Processed {line_count} lines.')

class QAClassifierTestDataset(Dataset):
    def __init__(self, Q, T, A, L):
        X = [' '.join(['[CLS]', Q[i], '[SEP]', T[i], '[SEP]']) for i in range(len(T))]
        self.L = []
        self.A = []
        self.input_ids = []
        self.attention_mask = []
        self.token_type_ids = []
        self.Q = []
        self.T = []

        for qid in tqdm(range(len(X)//10)):
            flag = True
            tempL, tempA, tempQ, tempT, tempInputIds, tempAttention, tempTokenType = [], [], [], [], [], [], []
            for i in range(qid*10, (qid+1)*10):
                x = X[i]
                tokenized_text = tokenizer.tokenize(x)
                # cut long paragraphs
                if len(tokenized_text) > maxLen:
                    # filter out examples with long comments+questions
                    tokenized_text = tokenized_text[:maxLen]
                    tokenized_text[maxLen-1] = '[SEP]'
                tempL.append(int(L[i]))
                tempA.append(A[i])
                tempQ.append(Q[i])
                tempT.append(T[i])
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
                tempInputIds.append(indexed_tokens)
                att = np.ones(len(indexed_tokens))
                att[eos2+1:] = 0
                types = np.ones(len(indexed_tokens))
                types[:eos1+1] = 0

                tempAttention.append(att)
                tempTokenType.append(types)
            if flag:
                self.L.extend(tempL)
                self.A.extend(tempA)
                self.Q.extend(tempQ)
                self.T.extend(tempT)
                self.input_ids.extend(tempInputIds)
                self.attention_mask.extend(tempAttention)
                self.token_type_ids.extend(tempTokenType)
                
        print('loaded '+str(len(self.input_ids))+' lines.')
    def __len__(self):
        return len(self.L)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx], self.L[idx], self.A[idx], self.Q[idx], self.T[idx]

def collate_fn(batch):
    input_ids, attention_mask, token_type_ids, L, A, Q, T = zip(*batch)
    return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(token_type_ids), torch.LongTensor(L), A, Q, T

    
eval_batch_size = 10
adam_epsilon = 1e-8
learning_rate = 5e-5 
def test(eval_dataset, model):      

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, sampler=eval_sampler, batch_size=eval_batch_size)  
    # Eval!
    print("***** Running evaluation *****")
    print("  Num examples = %d" % len(eval_dataset))
    print("  Batch size = %d" % eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    correct = 0
    numtop1, numtop5, numtop10 = 0, 0, 0
    total = 0
    bert_clean_result = []
    bert_pipeline_result = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        A = batch[4]
        Q = batch[5]
        Comment = batch[6]
        batch = tuple(t.to(device) for t in batch[:4])
        with torch.no_grad():
            inputs = {'input_ids':  batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        batch_res = logits.detach().cpu().numpy()
        batch_res = batch_res[:,1]
        labels = batch[3].detach().cpu().numpy()
#         correct += np.sum(labels == batch_res)
        topidx = np.argmax(batch_res)
        correct += 1 if topidx == 0 and labels[0] == 1 else 0
        if topidx < 1 and labels[0] == 1:
            numtop1 += 1
        if topidx < 5 and labels[0] == 1:
            numtop5 += 1
        if topidx < 10 and labels[0] == 1:
            numtop10 += 1
        total += 1
        # if total % 500 == 0:
        #     print('current #correct', correct, ', current #total', total)
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        #get clean test set for xlnet
        if topidx == 0 and labels[0] == 1:
            bert_clean_result.append([Q[0], A[0], Comment[topidx]])
        #get pipeline test set for xlnet
        bert_pipeline_result.append([Q[0], A[0], Comment[topidx]])
    print("correct:", correct)
    print("total:", total)
    print("acc:", correct/total)
    print("recall at 1:", numtop1/total)
    print("recall at 5:", numtop5/total)
    print("recall at 10:", numtop10/total)
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    return preds, bert_clean_result, bert_pipeline_result

test_dataset = QAClassifierTestDataset(Q, T, A, L)

preds, clean_result, pipeline_result = test(test_dataset, model)

col = ['QUESTION', 'ANSWER', 'COMMENT']
import csv

with open('./Cache/bert_pipeline_result.csv', mode='w') as output_file:
    csv_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(col)
    for row in pipeline_result:
        csv_writer.writerow(row)

import pandas as pd 
import json

# output json file

# [{'title': '1', 'paragraphs': [ 
#     {
#         'context': 'comment',
#         'qas': [{
#             'answers':[{'answer_start': 'char_level_position', 'text': 'answer'}],
#             'question': 'question',
#             'id': '1'
#         },{
#             'answers':[{'answer_start': 'char_level_position', 'text': 'answer'}],
#             'question': 'question',
#             'id': '2'
#         }]
#     },{}
# ]}, {}]

qid = 80000
count = 0
invalid_answer_count = 0
def process(fileName, data, qid):
    f = open(fileName)
    csv_reader = csv.reader(f, delimiter=',')
    
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            # print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            assert len(row) == 3
            question, answer, context = row[0], row[1], row[2]
            qas = []
            answer_start = context.find(answer)
            # answer_offset = related.find(answer)
            # answer_start = answer_offset if answer_offset == -1 else related_offset + answer_offset
            if answer_start == -1:
                global invalid_answer_count
                invalid_answer_count+=1

            qas.append({'answers':[{'answer_start': answer_start, 'text': answer}],
                    'question': question,
                    'id': str(qid)})
            qid+=1
            data.append({'context': context, 'qas': qas})
    return qid

test_data = []
fileName = './Cache/bert_pipeline_result.csv'
qid = process(fileName, test_data, qid)
test_result = {'data': [{'title': 'test_corpus', 'paragraphs': test_data}]}
# print(invalid_answer_count, qid-80000)

with open("./Cache/xlnet_pipeline.json", "w") as write_file:
    json.dump(test_result, write_file)
