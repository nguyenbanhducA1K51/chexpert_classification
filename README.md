
# CheXpert classification

## Purpose
This repository attempts to classifying 5 classes of CheXpert dataset (Cardiomegaly, Endema, Atelectasis, Pleural Effusion) using different combination of training techniques.
## Dataset
CheXpert is a extensive collection of chest X-rays used for automating chest X-ray analysis. This dataset includes uncertainty labels and sets evaluated by radiologists, serving as a benchmark for automated interpretations.
To download the dataset, visit this kaggle website: https://www.kaggle.com/datasets/mimsadiislam/chexpert
## Installation
Clone the repository
```bash
git clone https://github.com/nguyenbanhducA1K51/chexpert_classification.git
```
cd to the subdirectory 
```bash
cd chexpert_classification/chexpert
```

and use the package manager [pip](https://pip.pypa.io/en/stable/) to install libary and package in file requirements.txt (recommend install in conda environment).

```bash
pip install -r requirements.txt
```

## Usage

- Open the chexpert/config/config.py file and change the value of following variable:
"project_path": "path where you clone the this repository"
"root_path": "path where you download your dataset"
"train_csv_path": "path to train csv file"
 "test_csv_path": "path to test csv file"
- To adjust other parameter, look at other variable in this config file and some explainations at conifg_instruction.py
- Run the training pipeline by
```bash
python3 __main__.py
```



