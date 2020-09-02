import pandas as pd 
import csv
import random
import json

# read in json file

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

qid = 0
count = 0
invalid_answer_count = 0
def process(pre, post, data, length, qid):
    for mid in range(0, length+1, 500):
        fileName = pre + str(mid) + post
        f = open(fileName)
        lines = f.readlines()
        i = 1
        while(i < len(lines)):
            line = lines[i]
            cols = line[:-1].split('\t')
            context = cols[1]
            qas = []
            i+=1
            if i == len(lines):
                print(fileName)
                break
            line = lines[i]
            cols = line[:-1].split('\t')
            # qa pairs
            while(cols[0]==''):
                if len(cols) < 5:
                    print(fileName, line, cols)
                related, question, answer = cols[2], cols[3], cols[4]
                firstN = 15 if len(related) > 15 else len(related)
                related_offset = context.find(related[:firstN])
                # assert related_offset != -1, context + 'SEPSEPSEPSEPSEPSEP' + related
                if related_offset == -1:

                    global count
                    count += 1
                else:
                    answer_start = context.find(answer, related_offset)
                    # answer_offset = related.find(answer)
                    # answer_start = answer_offset if answer_offset == -1 else related_offset + answer_offset
                    if answer_start == -1:
                        global invalid_answer_count
                        invalid_answer_count+=1
                    else:
                        qas.append({'answers':[{'answer_start': answer_start, 'text': answer}],
                                'question': question,
                                'id': str(qid)})
                        qid+=1
                i+=1
                if i == len(lines):
                    break
                line = lines[i]
                cols = line[:-1].split('\t')
            data.append({'context': context, 'qas': qas})
    return qid

pre = './Cache/qaFromComments_train_'
post = '.csv'

train_data = []
qid = process(pre, post, train_data, 19000, qid)
train_result = {'data': [{'title': 'train_corpus', 'paragraphs': train_data}]}
print(count, invalid_answer_count, qid)

pre = './Cache/qqaFromComments_val_'
val_data = []
qid = process(pre, post, val_data, 1000, qid)
val_result = {'data': [{'title': 'val_corpus', 'paragraphs': val_data}]}
print(count, invalid_answer_count, qid)

pre = './Cache/qaFromComments_test_'
test_data = []
qid = process(pre, post, test_data, 1000, qid)
test_result = {'data': [{'title': 'test_corpus', 'paragraphs': test_data}]}
print(count, invalid_answer_count, qid)



with open("./Cache/xlnet_train.json", "w") as write_file:
    json.dump(train_result, write_file)

with open("./Cache/xlnet_val.json", "w") as write_file:
    json.dump(val_result, write_file)

with open("./Cache/xlnet_test.json", "w") as write_file:
    json.dump(test_result, write_file)
