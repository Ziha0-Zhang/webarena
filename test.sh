#!/bin/bash

source ./setup_env.sh

uv run scripts/generate_test_data.py

# prepare the evaluation
# re-validate login information
mkdir -p ./.auth
uv run browser_env/auto_login.py


uv run run.py \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
  --test_start_idx 0 \
  --test_end_idx 1 \
  --model deepseek-chat\
  --result_dir ./my_results \
  --save_trace_enabled \
  --render



