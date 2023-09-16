set -e

export PROJECT_ROOT="/root/repo/chexpert_classification/chexpert"
for FOLD in {1..5}
do
PYTHONPATH="${PROJECT_ROOT}" \
python "${PROJECT_ROOT}"/run.py "${PROJECT_ROOT}"/config/config.yaml --fold=$FOLD
done
