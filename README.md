Based on PyTorch implementation of [Lung Nodule Classification using Deep Local-Global Networks](https://arxiv.org/abs/1904.10126)

## Model
1. Model determines malignancy (whether lung tumour is cancerous or not) of segmented lung nodule.
2. It uses Residual Block (`BasicBlock` in `resnet_attn.py`) to extract local features like nodule density and texture.
3. It uses Self Attention (`SelfAttn` in `resnet_attn.py`) to extract global features like shape and size (varies from 3 to 30 mm).
4. Gamma value determines the weightage of global features in final output of the model. This also is learned by the model during training.

## Installing pre-requested libraries
`pip3 install requiremnets.txt`

## Dataset
1. This uses pre-processed dataset which contains segemented lung nodules from LIDC-IDRI Dataset CT scans. This can be downloaded from [here](https://drive.google.com/file/d/19JMK_IeBFlEQAEt_nrWsJcHrdyHcZMhm/view?usp=sharing) 
2. Copy downloaded dataset into `dataset/` folder

## Code
1. Run `python3 experiments.py dataset/` to get results. Model will be saved automatically.
2. Original Implementation uses multiple folds but due to resource constraints current code runs only for one fold.

## Experiment with number of layers
We added more layers to Residual Block so that model learns local features better.

| Number of Layers |   AUC  | Accuracy | Precision | Recall |
| :--------------: | :----: | :------: | :-------: | :----: |
|        2         | 0.9456 |  0.9347  |   0.8947  | 0.8252 |
|        3         | 0.9420 |  0.9408  |   0.9189  | 0.8292 |
|        4         | 0.9631 |  0.9598  |   0.9268  | 0.9268 |

## Experiment with Gamma value
Gamma value is intialized to zero and model learns this value during training. We increased this value.

|   Gamma Value    |   AUC  | Accuracy | Precision | Recall |
| :--------------: | :----: | :------: | :-------: | :----: |
|        0         | 0.9456 |  0.9347  |   0.8947  | 0.8252 |
|      0.25        | 0.9360 |  0.9359  |   0.8809  | 0.9024 |
