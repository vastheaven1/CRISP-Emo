# CRISP-Emo

Implementation of CRISP-Emo for DEAP and DREAMER.

## Install

```bash
pip install -r requirements.txt
```

## Preprocess

```bash
python3 preprocess_data.py \
  --deap-root /path/to/DEAP \
  --dreamer-mat /path/to/DREAMER.mat \
  --output-root /path/to/crisp_emo_data
```

Download the GRAM-B checkpoint from <https://github.com/iiieeeve/Gram> and
build the training cache:

```bash
python3 build_teacher_cache.py \
  --data-root /path/to/crisp_emo_data \
  --gram-weights /path/to/gram-b.pth \
  --output /path/to/crisp_emo_data/teacher_cache/fold0_train.pt \
  --fold 0 --device cuda:0
```

## Train

```bash
python3 train.py \
  --data-root /path/to/crisp_emo_data \
  --teacher-cache /path/to/crisp_emo_data/teacher_cache/fold0_train.pt \
  --results-root /path/to/results \
  --run-name crisp_emo_seed0 \
  --seed 0 --fold 0 --device cuda:0
```

## Evaluate

```bash
python3 evaluate_test.py \
  --run /path/to/results/RUN_DIRECTORY \
  --device cuda:0
```

## Test

```bash
pytest -q
```

