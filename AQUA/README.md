## AQUA Dataset

The AQUA Dataset is built on top of the [SemArt](http://noagarciad.com/semart/) dataset. 
Images for the AQUA dataset are obtained by downloading SemArt, whereas the question-answer annotations can be found in this repository.

<p align="center">
  <img width="460" src="https://github.com/noagarcia/ArtVQA/blob/master/images/examples.png">
</p>


### Download
- Painting images from SemArt dataset: [download](https://researchdata.aston.ac.uk/380/1/SemArt.zip)
- AQUA QA pairs in json format: [download](https://github.com/noagarcia/ArtVQA/blob/master/AQUA/)

### Annotations format

The QA pairs are split into 3 json files: 
[training](https://github.com/noagarcia/ArtVQA/blob/master/AQUA/train.json),
[validation](https://github.com/noagarcia/ArtVQA/blob/master/AQUA/val.json),
[evaluation](https://github.com/noagarcia/ArtVQA/blob/master/AQUA/test.json).

Each json file contains the following fields:
- `image`: image filename as in SemArt dataset.
- `question`: question 
- `answer`: answer
- `need_external_knowledge`: Whether the question requires external knowledge to be answered. `True` for QAs generated from comments and `False` for QAs generated from paintings.

### Statistics

There are 69,812 samples in the training set, 5,124 samples in the validation set and 4,912 QA samples in the test set.


