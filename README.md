
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

- Open the ../chexpert/config/config.yaml file and change the value of following variable:
"project_path": "path where you clone the this repository"
"data_path": "path where you download your dataset"
" process_train": "path that will save train/val  csv file after process them"
 "process_test": "path that will save train/test csv file after process them"

- First , run the preprocess file at "..chexpert/preprocess.py" 
```bash
python3 preprocess.py
```

- Run the training pipeline by
```bash
bash chexpert/scripts/train.sh
```



