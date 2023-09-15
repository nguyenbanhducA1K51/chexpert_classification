set -e

export PROJECT_ROOT="/root/repo/chexpert_classification/chexpert"
for FOLD in {1..5}
do
PYTHONPATH="${PROJECT_ROOT}" \
OMP_NUM_THREADS=1 \
python "${PROJECT_ROOT}"/run.py "${PROJECT_ROOT}"/configs/config.yaml --fold=$FOLD
done
