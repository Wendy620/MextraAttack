#!/bin/bash
python run_attack.py \
  --ehr_data ./data/general_30.json \
  --memory_data ./data/500_solution.json \
  --llm_model meta-llama/Llama-2-7b-hf \
  --backend amem \
  --out_dir ./outputs/amem
  
  python run_attack.py \
  --ehr_data ./data/general_30.json \
  --memory_data ./data/500_solution.json \
  --llm_model lmsys/vicuna-7b-v1.5 \
  --backend amem \
  --out_dir ./outputs/amem

python evaluation.py --input_dir ./outputs/amem --out_csv ./outputs/amem/summary.csv


