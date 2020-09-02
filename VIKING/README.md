## VIKING Baseline

### Overview

Our VIKING baseline consists on a three module model:

![model](https://github.com/noagarcia/ArtVQA/blob/master/images/model.png?raw=true)


1. [Modality selector](https://github.com/noagarcia/ArtVQA/blob/master/VIKING/modality_selector)
2. [Visual QA branch](https://github.com/noagarcia/ArtVQA/blob/master/VIKING/visual_branch)
3. [Knowledge QA branch](https://github.com/noagarcia/ArtVQA/blob/master/VIKING/knowledge_branch)

### Accuracy

- The `modality selector` accuracy is 99.6%.

- The `Visual QA branch` accuracy is 77.7%. That is 1,000 exact matches over 1,286 samples classified 
as *not external knowledge requiered* by the `Modality Selector`.

- The `Knowledge QA branch` accuracy is 47.6%. That is 1,726 exact matches over 3,626 samples classified 
as *external knowledge requiered* by the `Modality Selector`.

- The **total accuracy** of the system is 55.5%, with 1,000 exact matches in the `Visual QA branch` and 
1,726 exact matches in the `Knowledge QA branch` over a total of 4,912 samples in the test set.