## Modality selector
The `Modality selector` predicts whether external knowledge is needed to answer a question given its corresponding painting. 

Code by [Zihua Liu](https://github.com/Zihua-Liu). Maintenance by [Noa Garcia](https://github.com/noagarcia).


### Dependencies

```bash
pip install bert-serving-server
pip install bert-serving-client
conda install scikit-learn
```


### Feature Extraction

- **Question features**. We use pretrained BERT. First download pretrained model:

```bash
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip
unzip uncased_L-24_H-1024_A-16.zip
rm uncased_L-24_H-1024_A-16.zip
```

Then, run BERT service in the background:

```bash
bert-serving-start -model_dir ./uncased_L-24_H-1024_A-16/ -num_worker=1
```

- **Painting features**. We use pretrained ResNet-152. The extracted features for the all the paintings within SemArt dataset can be download from here:

```bash
wget https://semart.s3.amazonaws.com/image_features.json
```

- **Features processing**. Run feature extraction code to extract features and the extracted features will be saved to `Cache` folder:

```bash
python features.py
```


### Modality Selector

After feature extraction is done, we can then train the classifier and test its performance. First, make sure `train/val/test.npy` files are in `Cache` folder. Then, run:

```
python classifier.py
```

It will train the classifier on training set, validation its performance on validation set and predict result on test set. 
The prediction result of test set is stored in `Results/need_external_knowledge.json` and `Results/not_need_external_knowledge.json`.
