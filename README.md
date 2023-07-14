# Self-regulating Prompts: Foundational Model Adaptation without Forgetting [ICCV 2023]



> [**Self-regulating Prompts: Foundational Model Adaptation without Forgetting**](https://arxiv.org/abs/2307.06948)<br>
> [Muhammad Uzair Khattak*](https://muzairkhattak.github.io/), [Syed Talal Wasim*](https://talalwasim.github.io), [Muzammal Naseer](https://scholar.google.com/citations?user=tM9xKA8AAAAJ&hl=en&oi=ao), [Salman Khan](https://salman-h-khan.github.io/), [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en)

*Joint first authors

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2307.06948)
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://muzairkhattak.github.io/PromptSRC/)
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://drive.google.com/file/d/1d14q8hhAl6qGsiPYpNIVfShMCulVJSUa/view?usp=sharing)


Official implementation of the paper "[Self-regulating Prompts: Foundational Model Adaptation without Forgetting](https://arxiv.org/abs/2307.06948)".

<hr />

# :rocket: News
* **(July 14, 2023)**
  * Our work is accepted to ICCV 2023! :tada:
* **(July 12, 2023)**
  * Pre-trained models and evaluation codes for reproducing PromptSRC official benchmark results are released.
  * Training codes for [PromptSRC](configs/trainers/PromptSRC) are released.
  * This repository also supports [MaPle (CVPR'23)](configs/trainers/MaPLe),
[CoOp (IJCV'22)](configs/trainers/CoOp), [Co-CoOp (CVPR'22)](configs/trainers/CoCoOp) 
architectures.
<hr />

## Highlights

![main figure](docs/main_figure.png)
> <p align="justify"> <b> <span style="color: blue;">Left</span></b>:
> Existing prompt learning approaches for foundational Vision-Language models like CLIP rely on task-specific objectives that restrict
> prompt learning to learn a feature space suitable only for downstream tasks and 
> consequently lose the generalized knowledge of CLIP (shown in <span style="color: purple;">purple</span></b>). 
> Our self-regulating framework explicitly guides the training trajectory of prompts
> towards the closest point between two optimal solution manifolds (solid line) to 
> learn task-specific representations while also retaining generalized CLIP knowledge
> (shown in <span style="color: green;">green</span>). <b><span style="color: blue;">Middle</span></b>: Averaged 
> across 11 image recognition datasets, PromptSRC surpasses existing methods on the
> base-to-novel generalization setting. <b><span style="color: blue;">Right</span></b>: We evaluate
> our approach on four diverse image recognition benchmarks for CLIP and show 
> consistent gains over previous state-of-the-art approaches. </p>





> **<p align="justify"> Abstract:** *Prompt learning has emerged as an efficient alternative 
> for fine-tuning foundational models, such as CLIP, for various downstream tasks.
> Conventionally trained using the task-specific objective, i.e., cross-entropy loss, 
> prompts tend to overfit downstream data distributions and find it challenging to capture
> task-agnostic general features from the frozen CLIP. This leads to the loss of the model's 
> original generalization capability. To address this issue, our work introduces a 
> self-regularization framework for prompting called PromptSRC (Prompting with Self-regulating 
> Constraints). PromptSRC guides the prompts to optimize for both task-specific and task-agnostic
> general representations using a three-pronged approach by: (a) regulating {prompted}
> representations via mutual agreement maximization with the frozen model, (b) regulating 
> with self-ensemble of prompts over the training trajectory to encode their complementary
> strengths, and (c) regulating with textual diversity to mitigate sample diversity imbalance
> with the visual branch. To the best of our knowledge, this is the first regularization 
> framework for prompt learning that avoids overfitting by jointly attending to pre-trained
> model features, the training trajectory during prompting, and the textual diversity. 
> PromptSRC explicitly steers the prompts to learn a representation space that maximizes
> performance on downstream tasks without compromising CLIP generalization. We perform
> experiments on 4 benchmarks where PromptSRC performs favorably well compared
> to the existing methods. Our code and pre-trained models are publicly available.* </p>

## Regularization Framework for Prompt Learning

We propose PromptSRC (Prompting with Self-regulating Constraints) which steers the prompts to learn a representation space that maximizes performance on downstream tasks without compromising CLIP generalization.

**Key components of PromptSRC:**
1) **Mutual agreement maximization:** PromptSRC explicitly guides the prompts to jointly acquire both <i>task-specific knowledge</i> and <i>task-agnostic generalized knowledge</i> by maximizing the mutual agreement between prompted and features of the frozen VL model.
2) **Gaussian weighted prompt aggregation:** We propose a weighted self-ensembling strategy for prompts over the training trajectory that captures complementary features and enhances their generalization abilities.
3) **Textual diversity:** PromptSRC regulates prompts with textual diversity to mitigate sample diversity imbalance compared to the visual branch during training.


## :ballot_box_with_check: Supported Methods

| Method                    | Paper                                         |                             Configs                             |        Training Scripts         |
|---------------------------|:----------------------------------------------|:---------------------------------------------------------------:|:-------------------------------:|
| PromptSRC                 | [arXiv](https://arxiv.org/abs/2307.06948)                                     |                    [link](configs/trainers/PromptSRC/)                    |    [link](scripts/promptsrc)    |
| Independent V-L Prompting | -                                             | [link](configs/trainers/IVLP/) | [link](scripts/independent-vlp) |
  | MaPLe                     | [CVPR 2023](https://arxiv.org/abs/2210.03117) |                  [link](configs/trainers/CoOp)                  |      [link](scripts/maple)      |
| CoOp                      | [IJCV 2022](https://arxiv.org/abs/2109.01134) |                  [link](configs/trainers/CoOp)                  |      [link](scripts/coop)       |
| Co-CoOp                   | [CVPR 2022](https://arxiv.org/abs/2203.05557) |                 [link](configs/trainers/CoCoOp)                 |     [link](scripts/cocoop)      |

<hr />

## Results
Results reported below show accuracy for base and novel classes for across 11 recognition datasets averaged over 3 seeds.


### Effectiveness of PromptSRC in comparison with baseline Independent V-L Prompting
PromptSRC effectively maximizes supervised task performance (base classes) without compromising on CLIP's original generalization towards new unseen tasks (novel classes).

| Name                                                                            | Base Acc. | Novel Acc. |    HM     | Epochs |  
|---------------------------------------------------------------------------------|:---------:|:----------:|:---------:|:------:|
| CLIP  |   69.34   |   74.22    |   71.70   |   -    |  
| Independent V-L Prompting |   84.21   |   71.79    |   77.51   |   20   | 
| PromptSRC (ours) | **84.26** | **76.10**  | **79.97** |   20   | 



### PromptSRC in comparison with existing state-of-the-art

| Name                                       | Base Acc. | Novel Acc. |    HM     | Epochs | 
|--------------------------------------------|:---------:|:----------:|:---------:|:------:|
| [CLIP](https://arxiv.org/abs/2103.00020)   |   69.34   |   74.22    |   71.70   |   -    |  
| [CoOp](https://arxiv.org/abs/2109.01134)   |   82.69   |   63.22    |   71.66   |  200   | 
| [CoCoOp](https://arxiv.org/abs/2203.05557) |   80.47   |   71.69    |   75.83   |   10   | 
| [ProDA](https://arxiv.org/abs/2205.03340)  |   81.56   |   75.83    |   76.65   |  100   | 
| [MaPLe](https://arxiv.org/abs/2210.03117)                           |   82.28   | 75.14  | 78.55 |   5    |
| [PromptSRC (ours)](https://arxiv.org/abs/2307.06948)                       | **84.26** | **76.10**  | **79.97** |   20   |  

## Installation 
For installation and other package requirements, please follow the instructions detailed in [INSTALL.md](docs/INSTALL.md). 

## Data Preparation
Please follow the instructions at [DATASETS.md](docs/DATASETS.md) to prepare all datasets.

## Model Zoo

### Vision-Language prompting methods
| Name  (configs)                                                                       |                                                             Model checkpoints                                                             |
|---------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------:|
| [Independent V-L Prompting](configs/trainers/IVLP/vit_b16_c2_ep20_batch4_4+4ctx.yaml) | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/syed_wasim_mbzuai_ac_ae/EuIwh-yMh_JBqB2Y_o8Jl14BPDKDRHC0JBPE1BugIeZiSQ?e=AJ8MhY) |
| [PromptSRC](configs/trainers/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx.yaml)            | [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/syed_wasim_mbzuai_ac_ae/EqFXPs2Zl9pKp39w3SqlR7QBDACTv-AgCXH6_cGflrUFwg?e=l33EBA) |


## Evaluation 
Please refer to the [EVAL.md](docs/EVAL.md) for detailed instructions on using the evaluation scripts and reproducing the official results using our pre-trained models.

## Training 
Please refer to the [TRAIN.md](docs/TRAIN.md) for detailed instructions on training PromptSRC and IVLP baseline from scratch.


<hr />

## Citation
If you find our work, this repository, or pretrained models useful, please consider giving a star :star: and citation.
```bibtex
@article{khattak2023PromptSRC,
    title={Self-regulating Prompts: Foundational Model Adaptation without Forgetting},
    author={khattak, Muhammad Uzair and Wasim, Syed Talal and Muzzamal, Naseer and Khan, Salman and Yang, Ming-Hsuan and Khan, Fahad Shahbaz},
    journal={arXiv:2307.06948},
    year={2023}
}
```

## Contact
If you have any questions, please create an issue on this repository or contact at uzair.khattak@mbzuai.ac.ae or syed.wasim@mbzuai.ac.ae.


## Acknowledgements

Our code is based on [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning), along with [Co-CoOp and CoOp](https://github.com/KaiyangZhou/CoOp) repository. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.

