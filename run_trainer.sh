#! /bin/bash

python trainer_feedback.py --task meta_learning --retry_limit 3 --model_name meta-llama/Meta-Llama-3-8B-Instruct --init_kl_coef 0.075 --starting_step 0 --batch_size 32 --use_peft