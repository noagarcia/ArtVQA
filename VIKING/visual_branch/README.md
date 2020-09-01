## Visual QA branch selector
The `visual QA branch` predicts an answer for a sample classified as not external knowledge is needed.

[iQAN](https://github.com/yikang-li/iQAN) is the VQA model we use in our pipeline to answer visual-related question. 
It takes the question which external knowledge classifier predicts that no external knowledge is needed as well as the 
painting as input, and predicts an answer to the given question.

See list of requierements [here](https://github.com/Zihua-Liu/QA_Pipeline_SemArt/blob/master/iQAN/requirements.txt).


### Prepare Data

In order to train and evaluate the performance of the model, first prepare the training, valiadation and test. 

- Copy training and validation set from to the target folder:

```bash
cp ../../AQUA/train.json ./data/SemArt/extract/arch,resnet152_size,448/
cp ../../AQUA/val.json ./data/SemArt/extract/arch,resnet152_size,448/
```

- The test set is those questions which `modality selector` predicts that no external knowledge is needed, the file
`test_not_need_kg.json` contains the test set for this module.

```bash
cp ../modality_selector/Results/test_not_need_kg.json ./data/SemArt/extract/arch,resnet152_size,448/
```

### Prepare Features for Painings

We use pretrained ResNet features to represent paintings. The features and the index of the paintings can be downloaded from here:

```
wget https://semart.s3.amazonaws.com/all.hdf5 -P ./data/SemArt/extract/arch,resnet152_size,448/
wget https://semart.s3.amazonaws.com/all.txt -P ./data/SemArt/extract/arch,resnet152_size,448/
```

The `hdf5` file is about 30GB and may take some time to download.

### Run the model

- Training:

```bash
CUDA_VISIBLE_DEVICES=0 python train_dual_model.py --path_opt options/dual_model/dual_model_MUTAN_skipthought.yaml --dir_logs logs/dual_model/iQAN_Mutan_skipthought_dual_training/ --share_embeddings -b 8
```

- Evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python train_dual_model.py --path_opt options/dual_model/dual_model_MUTAN_skipthought.yaml --dir_logs logs/dual_model/iQAN_Mutan_skipthought_dual_training/ --resume best -e --share_embeddings -b 8
```

The model will load a checkpoint which has best validation performance and do the predicion on test set. The prediction result can be seen in `logs/dual_model/iQAN_Mutan_skipthought_dual_training/evaluate`.
