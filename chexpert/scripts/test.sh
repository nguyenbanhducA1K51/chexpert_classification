set -e

export PROJECT_ROOT="/root/repo/chexpert_classification/chexpert"
python "${PROJECT_ROOT}"/run.py "${PROJECT_ROOT}"/config/config.yaml --mode test 

