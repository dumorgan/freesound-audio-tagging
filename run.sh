#!/bin/bash -e

# MODEL=mobilenet-v1
MODEL=xception


python runner.py --mode train --model $MODEL  --class_map_path class_map.csv --train_clip_dir data --train_csv_path train_curated.csv --train_dir train/

python runner.py --mode eval  --model $MODEL --class_map_path class_map.csv --eval_clip_dir data --eval_csv_path validation_curated.csv --train_dir $(pwd)/train/ --eval_dir $(pwd)/validation/

sudo shutdown now
