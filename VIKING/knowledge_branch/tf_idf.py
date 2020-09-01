import math
import numpy as np
import json
from scipy.spatial import distance
import copy
import os
    
def term_freq (document):
    dict_tf = dict()
    total_word = 0
    for word in document:
        if word.strip() != "": 
            total_word += 1
            if word not in dict_tf:
                dict_tf[word] = 1
            else:
                dict_tf[word] += 1
    for word in dict_tf:
        dict_tf[word] = dict_tf[word]/float(total_word)
    return dict_tf

def inverse_doc_freq (all_documents):
    dict_idf = dict()
    total_documents = len(all_documents)
    for document in all_documents:
        set_words = set()
        for word in document:
            if word.strip() != "":
                set_words.add(word)
        
        for word in set_words:        
            if word not in dict_idf:
                dict_idf[word] = 1
            else:
                dict_idf[word] += 1
    for word in dict_idf:
        dict_idf[word] = 1.0 + math.log(float(total_documents)/dict_idf[word])
    return dict_idf   
  
def cosine_sim (vector_q, vector_d):
    dot_product = np.dot(vector_q, vector_d)
    vector_q_norm = np.sqrt(np.dot(vector_q, vector_q))
    vector_d_norm = np.sqrt(np.dot(vector_d, vector_d))
    if vector_q_norm == 0.0:
        return 0.0
    elif vector_d_norm == 0.0:
        return 0.0
    else:
        return dot_product/(vector_q_norm*vector_d_norm)

def inverse_euclidean (vector_q, vector_d):
    dst = distance.euclidean(vector_q, vector_d)
    return 1.0 / dst

def tf_idf (question, all_documents, idf_dict, tf_dict_d_list, mode = 'cosine_sim'):

    tf_dict_q = term_freq(question)
    
    vector_q = []
    for word in question:
        if word.strip() != "":
            tf = tf_dict_q[word]
            if word in idf_dict:
                idf = idf_dict[word]
            else:
                idf = 0.0
            vector_q.append(tf*idf)

    score_array = []
    
    # for document in all_documents:
        
    #     tf_dict_d = term_freq(document)
    for tf_dict_d in tf_dict_d_list:
        vector_d = []
        for word in question:
            if word.strip() != "":
                if word in tf_dict_d:
                    tf = tf_dict_d[word]
                else:
                    tf = 0.0
                    
                if word in idf_dict:
                    idf = idf_dict[word]
                else:
                    idf = 0.0
                
                vector_d.append(tf*idf)
    
        if mode == 'cosine_sim':
            score = cosine_sim(vector_q, vector_d)
        elif mode == 'inverse_euclidean':
            score = inverse_euclidean(vector_q, vector_d)
            
        score_array.append(score) 
    return np.array(score_array)

def predict(question, documents, idf_dict, tf_dict_d_list):
    scores = tf_idf(question, documents, idf_dict, tf_dict_d_list)
    result = list(scores.argsort())
    result.reverse()
    return result[0:10]


def n_gram_wrapper(words, max_n):
    result = copy.deepcopy(words)
    for n in range(2, max_n + 1):
        for i in range(len(words) - n + 1):
            result.append("".join(words[i:i + n]))
    return result



if __name__ == "__main__":

    cache_dir = "Cache/"
    with open(os.path.join(cache_dir, "preprocessed_test_need_kg.json")) as f:
        questions = json.load(f)
    with open(os.path.join(cache_dir, "preprocessed_comments.json")) as f:
        comments = json.load(f)
    documents = [n_gram_wrapper(item["filtered_comment"], 3) for item in comments]

    idf_dict = inverse_doc_freq(documents)
    tf_dict_d_list = []
    for document in documents:
        tf_dict_d_list.append(term_freq(document))

    result = []
    cnt = 0
    acc1 = 0
    acc3 = 0
    acc5 = 0
    acc10 = 0
    for item in questions:
        label = item["image"]
        pred_indexes = predict(n_gram_wrapper(item["filtered_question"], 3), documents, idf_dict, tf_dict_d_list)
        prediction = []
        for index in pred_indexes:
            prediction.append(comments[index]["image"])

        cnt += 1
        if label in prediction[:1]:
            acc1 += 1
        if label in prediction[:3]:
            acc3 += 1
        if label in prediction[:5]:
            acc5 += 1
        if label in prediction[:10]:
            acc10 += 1
        print("[{}/{}]: Acc1: {}, Acc3: {}, Acc5: {}, Acc10: {}".format(cnt, len(questions), float(acc1) / cnt, float(acc3) / cnt , float(acc5) / cnt, float(acc10) / cnt,))
        item["comments_prediction_top10"] = prediction
        result.append(item)

    with open(os.path.join(cache_dir, "predicted_kg_comment_top10_tfidf.json"), "w") as f:
        json.dump(result, f, indent = 4)










