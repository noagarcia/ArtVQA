import pandas as pd 
import csv
import random
import json

#construct the dictionary (key: imgId, val: comment)
img2comment = {}
train_file = open("./comments/semart_train.csv", encoding='mac_roman')
csv_reader = csv.reader(train_file, delimiter='\t')
line_count = 0
for row in csv_reader:
    if line_count == 0:
        print(row[0], row[1])
        line_count += 1
    else:
        img2comment[row[0]] = row[1]
        line_count += 1
print(f'Processed {line_count} lines.')
val_file = open("./comments/semart_val.csv", encoding='mac_roman')
csv_reader = csv.reader(val_file, delimiter='\t')
line_count = 0
for row in csv_reader:
    if line_count == 0:
        print(row[0], row[1])
        line_count += 1
    else:
        img2comment[row[0]] = row[1]
        line_count += 1
print(f'Processed {line_count} lines.')
test_file = open("./comments/semart_test.csv", encoding='mac_roman')
csv_reader = csv.reader(test_file, delimiter='\t')
line_count = 0
for row in csv_reader:
    if line_count == 0:
        print(row[0], row[1])
        line_count += 1
    else:
        img2comment[row[0]] = row[1]
        line_count += 1
print(f'Processed {line_count} lines.')

# read in json file
col = ['QUESTION', 'COMMENT', 'ANSWER', 'LABEL']
test_data = []

with open('./Cache/predicted_kg_comment_top10_tfidf.json') as json_file:
    content = json.load(json_file)
    count = 0
    correct = 0
    for example in content:
        question = example['question']
        positive = example['image']
        answer = example['answer']
        if positive in example['comments_prediction_top10']:
            test_data.append([question, img2comment[positive], answer, 1])
            for img in example['comments_prediction_top10']:
                if img != positive:
                    test_data.append([question, img2comment[img], answer, 0])
            correct += 1
        else:
            for img in example['comments_prediction_top10']:
                test_data.append([question, img2comment[img], answer, 0])
        count += 1
    print('last stage acc:', correct/count, ', #correct:', correct, ', #total', count)
test_df = pd.DataFrame(data=test_data, columns=col)
test_df.to_csv('./Cache/bert_predicted_kg.csv', index=False)

