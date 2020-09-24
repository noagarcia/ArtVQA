## Visual QA branch selector
The `visual QA branch` predicts an answer for a sample classified as not external knowledge is needed.

[iQAN](https://github.com/yikang-li/iQAN) is the VQA model we use in our pipeline to answer visual-related question. 
It takes the question which external knowledge classifier predicts that no external knowledge is needed as well as the 
painting as input, and predicts an answer to the given question.

### Dependencies
```bash
docker build --build-arg USERNAME=$(whoami) --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t image_name_here .
docker run --rm -it --init --volume="/path/to/ArtQA_root_dir:/workspace" image_name_here bash
```

### Prepare Data
Dataset files are specified in `conf/params.yml`.

```yml
dataset:
    train: ../../AQUA/train.json
    val: ../../AQUA/val.json
    test: ../../AQUA/test.json
```

First compile vocabulary files from the training data. 

```bash
python vqa/datasets/vqa_processed.py conf/params.yml
```

- The test set `AQUA/test.json` includes both visual questions and those that need external knowledge. If you want to test a model only on visual questions, extract questions with `"need_external_knowledge": false` and save in another file. Then, specify the path to the file in `conf/param.yml`.

### Prepare Features for Painings

We use pretrained ResNet features to represent paintings. The features and the index of the paintings can be downloaded from here:

```
wget https://semart-images.s3-ap-northeast-1.amazonaws.com/VIKING/all.hdf5 -P ./data/SemArt/extract/arch,resnet152_size,448/
wget https://semart-images.s3-ap-northeast-1.amazonaws.com/VIKING/all.txt -P ./data/SemArt/extract/arch,resnet152_size,448/
```

The `hdf5` file is about 30GB and may take some time to download.

### Run the model

- Training:

```bash
python train.py conf/params.yml
```

- Logging with neptune (optional)

[Neptune](https://neptune.ai/) is an experiment tracking tool.
Specify neptune project and experiment names in `conf/params.yml` and run with `--neptlog` flag.

```bash
python train.py conf/params.yml --neptlog
```

- Evaluation:

```bash
python evaluate.py directory/of/model-checkpoint/
```

The model will load a checkpoint which has best validation performance and do the predicion on test set. The prediction result can be seen in `directory/of/model-checkpoint/evaluate`.
