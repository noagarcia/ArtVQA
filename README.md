## Visual Question Answering on Art

Repository for the [paper](https://arxiv.org/abs/2008.12520) *A Dataset and Baselines for Visual Question Answering on Art*, 
published at [VISART](https://visarts.eu/) workshop at [ECCV 2020](https://eccv2020.eu/).

Watch the paper introduction [video](https://www.youtube.com/watch?v=I78SoOkH3dM&t=116s).

<p align="center">
  <img width="460" src="https://github.com/noagarcia/ArtVQA/blob/master/images/examples.png">
</p>

Answering questions related to paintings implies the understanding of not only the visual information that is shown in the picture, 
but also the contextual knowledge that is acquired through the study of the history of art. We introduce a dataset and baselines to explore this challenging task. Specifically, in this repository you can find:
- **AQUA Dataset**. The question-answer (QA)
pairs are automatically generated using state-of-the-art question generation methods based on paintings and comments provided in an existing
art understanding dataset.
- **VIKING Baselines**. We present a two-branch model as baseline for the task of visual question answering on art. 
In VIKING the visual and knowledge questions are handled independently. 

### AQUA Dataset

The AQUA Dataset is built on top of the [SemArt](http://noagarciad.com/semart/) dataset. 
Images for the AQUA dataset are obtained by downloading SemArt, whereas the question-answer annotations can be found in this repository.

##### Download
- Painting images from SemArt dataset: [downlowad](https://researchdata.aston.ac.uk/380/1/SemArt.zip)
- AQUA QA pairs in json format: [download](https://github.com/noagarcia/ArtVQA/blob/master/AQUA/)

##### Annotations format

The QA pairs are split into 3 json files: 
[training](https://github.com/noagarcia/ArtVQA/blob/master/AQUA/train.json),
[validation](https://github.com/noagarcia/ArtVQA/blob/master/AQUA/val.json),
[evaluation](https://github.com/noagarcia/ArtVQA/blob/master/AQUA/test.json).

Each json file contains the following fields:
- `image`: image filename as in SemArt dataset.
- `question`: question 
- `answer`: answer
- `need_external_knowledge`: Whether the question requires external knowledge to be answered. `True` for QAs generated from comments and `False` for QAs generated from paintings.

##### Statistics

There are 69,812 samples in the training set, 5,124 samples in the validation set and 4,912 QA samples in the test set.



### VIKING Baseline

##### Overview

Our VIKING baseline consists on a three module model.

![model](https://github.com/noagarcia/ArtVQA/blob/master/images/model.png?raw=true)


### Citation

If you find this code useful, please cite our work:
````
@InProceedings{garcia2020AQUA,
   author    = {Noa Garcia and Chentao Ye and Zihua Liu and Qingtao Hu and 
                Mayu Otani and Chenhui Chu and Yuta Nakashima and Teruko Mitamura},
   title     = {A Dataset and Baselines for Visual Question Answering on Art},
   booktitle = {Proceedings of the European Conference in Computer Vision Workshops},
   year      = {2020},
}
````

