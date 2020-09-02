## Visual Question Answering on Art

Repository for the [paper](https://arxiv.org/abs/2008.12520) *A Dataset and Baselines for Visual Question Answering on Art*, 
published at [VISART](https://visarts.eu/) workshop at [ECCV 2020](https://eccv2020.eu/).

Watch the paper introduction [video](https://www.youtube.com/watch?v=I78SoOkH3dM&t=116s).

<p align="center">
  <img width="460" src="https://github.com/noagarcia/ArtVQA/blob/master/images/examples.png">
</p>

Answering questions related to paintings implies the understanding of not only the visual information that is shown in the picture, 
but also the contextual knowledge that is acquired through the study of the history of art. We introduce a dataset and baselines to explore this challenging task. Specifically, in this repository you can find:
- [AQUA Dataset](https://github.com/noagarcia/ArtVQA/blob/master/AQUA/). The question-answer (QA)
pairs are automatically generated using state-of-the-art question generation methods based on paintings and comments provided in an existing
art understanding dataset.
- [VIKING Baseline](https://github.com/noagarcia/ArtVQA/blob/master/VIKING/). We present a two-branch model as baseline for the task of visual question answering on art. 
In VIKING the visual and knowledge questions are handled independently. 


### Citation

If you find our work useful, please us:
````
@InProceedings{garcia2020AQUA,
   author    = {Noa Garcia and Chentao Ye and Zihua Liu and Qingtao Hu and 
                Mayu Otani and Chenhui Chu and Yuta Nakashima and Teruko Mitamura},
   title     = {A Dataset and Baselines for Visual Question Answering on Art},
   booktitle = {Proceedings of the European Conference in Computer Vision Workshops},
   year      = {2020},
}
````

