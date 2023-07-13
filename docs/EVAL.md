# Evaluating and Reproducing PromptSRC Results

We provide bash scripts in [scripts/](../scripts) directory for evaluating PromptSRC and independent V-L prompting baseline using the provided pre-trained model checkpoints.


Make sure to update the `DATA` variable with dataset path in the script file and run the commands from the main directory `PromptSRC/`.
Below we provide the pre-trained models evaluation instructions for PromptSRC. The same instructions applies for reproducing results for the baseline *independent V-L prompting* and MaPLe.

## PromptSRC

#### (1) Base-to-Novel class generalization setting
The base-to-novel PromptSRC configuration is provided in config file at `configs/trainers/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx.yaml`. No hyper-parameters or other settings should be changed in the config file during evaluation of pre-trained models. 

We show an example to reproduce results for imagenet. Follow the instructions below to reproduce results using our pre-trained model weights:
* Download the zipped folder containing base-to-novel generalization pre-trained weights for a single dataset from this [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/syed_wasim_mbzuai_ac_ae/Em_3tkSj6T9AmhVjmzKTL3gBYNehhvfJl8ke2pU3U0nabA?e=9ecjQA). After unzipping, the directory should look like this:

```
imagenet
|–– base/
|   |–– seed1/
|   |–– seed2/
|   |–– seed3/
```

Now use the evaluation script `scripts/promptsrc/reproduce_base2novel_setting.sh` and run the commands below to calculate the results over 3 seeds:
```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# evaluate on base and novel classes for SEED1
bash scripts/promptsrc/reproduce_base2novel_setting.sh imagenet 1 /path/to/imagenet/weights/folder
# evaluate on base and novel classes for SEED2
bash scripts/promptsrc/reproduce_base2novel_setting.sh imagenet 2 /path/to/imagenet/weights/folder
# evaluate on base and novel classes for SEED3
bash scripts/promptsrc/reproduce_base2novel_setting.sh imagenet 3 /path/to/imagenet/weights/folder
```

This should evaluate and save the log files in `output/` directory. To obtain the averaged results, run:

```bash
# prints averaged results for base classes
python output/base2new/test_base/imagenet/shots_16/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx --test-log
# prints averaged results for novel classes
python output/base2new/test_new/imagenet/shots_16/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx --test-log
```
The same above steps can be repeated for other individual datasets by providing respective dataset name and checkpoints path.


#### (2) Cross-dataset and domain generalization setting
In cross-dataset and domain generalization setting, we first train PromptSRC on ImageNet-1k in few-shot manner with 16 shots for all 3 seeds and then evaluate the trained model directly on cross-datasets and out-of-distribution datasets.

We provide the instructions below to reproduce cross-datasets and domain generalization results using our pre-trained imagenet model weights for PromptSRC:
* Download the zipped folder containing pre-trained weights for imagenet from this [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/syed_wasim_mbzuai_ac_ae/Ekr9qF0cSaVDr0X6OlP2JAEBG1xjlTMjHNLc28g1SjwW-w?e=AA5ABi). After unzipping, the directory should look like this:

```
imagenet
|–– seed1/
|–– seed2/
|–– seed3/
```

Now use the evaluation script `scripts/promptsrc/reproduce_xd.sh` and run the commands below to calculate the results for food101 dataset over 3 seeds:
```bash
# Other possible dataset values for cross-datasets includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]
# possible dataset values for domain generalization benchmark includes [imagenetv2, imagenet_sketch, imagenet_a, imagenet_r]

# evaluate on given dataset for SEED1
bash scripts/promptsrc/reproduce_xd.sh food101 1 /path/to/imagenet/weights/folder
# evaluate on given dataset for SEED2
bash scripts/promptsrc/reproduce_xd.sh food101 2 /path/to/imagenet/weights/folder
# evaluate on given dataset for SEED3
bash scripts/promptsrc/reproduce_xd.sh food101 3 /path/to/imagenet/weights/folder
```

This should evaluate and save the log files in `output/` directory. To obtain the results averaged over 3 seeds, run:

```bash
# prints averaged results for food101 dataset
python parse_test_res.py output/evaluation/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx_cross_datasets_16shots/food101 --test-log
```

The same above steps can be repeated for other individual datasets by providing respective dataset name and checkpoints path.


#### (3) Few-shot setting
In this setting, PromptSRC is trained on all classes individual datasets with different few-shot splits (K = 1, 2, 4, 8, 16). The PromptSRC config for few-shot setting is available at: `configs/trainers/PromptSRC/vit_b16_c2_ep50_batch4_4+4ctx_few_shot.yaml`. 
Follow the instructions below to reproduce PromptSRC few-shot setting results using our pre-trained models:

Now use the evaluation script `scripts/promptsrc/reproduce_few_shot.sh` and run the commands below to calculate the results for imagenet dataset over 3 seeds:
```bash
# reproduce_few_shot.sh calculates results for all 3 seeds for a given K
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# evaluate on given dataset for K=1 shot
bash scripts/promptsrc/reproduce_few_shot.sh food101 1 /path/to/imagenet/weights/folder
# evaluate on given dataset for K=2 shot
bash scripts/promptsrc/reproduce_few_shot.sh food101 2 /path/to/imagenet/weights/folder
# evaluate on given dataset for K=4 shot
bash scripts/promptsrc/reproduce_few_shot.sh food101 4 /path/to/imagenet/weights/folder
# evaluate on given dataset for K=8 shot
bash scripts/promptsrc/reproduce_few_shot.sh food101 8 /path/to/imagenet/weights/folder
# evaluate on given dataset for K=16 shot
bash scripts/promptsrc/reproduce_few_shot.sh food101 16 /path/to/imagenet/weights/folder
```

This should evaluate and save the log files in `output/` directory. To obtain the results averaged over 3 seeds for all shots, run:

```bash
# prints averaged results for food101 dataset for K=1
python parse_test_res.py output/few_shot/food101/PromptSRC/vit_b16_c2_ep50_batch4_4+4ctx_few_shot_1shots/food101 --test-log
# prints averaged results for food101 dataset for K=2
python parse_test_res.py output/few_shot/food101/PromptSRC/vit_b16_c2_ep50_batch4_4+4ctx_few_shot_2shots/food101 --test-log
# prints averaged results for food101 dataset for K=4
python parse_test_res.py output/few_shot/food101/PromptSRC/vit_b16_c2_ep50_batch4_4+4ctx_few_shot_4shots/food101 --test-log
# prints averaged results for food101 dataset for K=8
python parse_test_res.py output/few_shot/food101/PromptSRC/vit_b16_c2_ep50_batch4_4+4ctx_few_shot_8shots/food101 --test-log
# prints averaged results for food101 dataset for K=16
python parse_test_res.py output/few_shot/food101/PromptSRC/vit_b16_c2_ep50_batch4_4+4ctx_few_shot_16shots/food101 --test-log
```

The same above steps can be repeated for other individual datasets by providing respective dataset name and checkpoints path.

<br>

## Training and Evaluating the independent V-L prompting baseline results

For IVLP baseline method, we provide its corresponding default configs and evaluation scripts as follows.

```
configs
|–– datasets/
|–– trainers/
|   |–– CoCoOp/
|   |–– CoOp/
|   |–– MaPLe/
|   |–– IVLP/
|   |–– PromptSRC/
```

```
scripts
|–– cocoop/
|–– coop/
|–– maple/
|–– independent-vlp/
|–– promptsrc/
```

Please use the corresponding config and script files and follow the same instructions as provided for PromptSRC in order to evaluate and reproduce results of IVLP baseline approach. The pretrained weights for IVLP baseline are provided [at this link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/syed_wasim_mbzuai_ac_ae/EuIwh-yMh_JBqB2Y_o8Jl14BPDKDRHC0JBPE1BugIeZiSQ?e=oJnJwy). 
This repository also supports using official [CoOp](CoOp.md) and [Co-CoOp](Co-CoOp.md) configs and models.