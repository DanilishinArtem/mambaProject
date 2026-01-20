# rm -rf /home/adanilishin/mambaProject/transformers_ssm_copy/tensorboard/model_mambapp_layer_12_hidden_1024_heads_0_train_copy_lr_1e-05_epochs_1_steps_2000
# python3 synthetic_exps/main.py --model "mambapp" --train_task "copy" --eval_task  "copy" --min_train_len 5 --max_train_len 20 --min_eval_len 15 --max_eval_len 20 --steps 2000 --heads 0 --n_gram 0 --context_len 44

# 'mamba', 'mambapp', 'linformer'

# rm -rf /home/adanilishin/mambaProject/transformers_ssm_copy/tensorboard/model_mambapp_layer_12_hidden_1024_heads_1_train_copy_lr_1e-05_epochs_1_steps_10000
# python3 synthetic_exps/main.py --train_batch_size 2 --state_dim 16 --context_len 220 --model "mambapp" --train_task "copy" --eval_task  "copy" --min_train_len 50 --max_train_len 100 --min_eval_len 15 --max_eval_len 200 --steps 10000 --heads 1  --epochs 1 --layers 12

# python3 synthetic_exps/main.py --model "T_alibi" --train_task "copy" --eval_task  "copy" --min_train_len 5 --max_train_len 50 --min_eval_len 15 --max_eval_len 100 --steps 2000 --heads 1 --n_gram 0 --context_len 201
# python3 synthetic_exps/main.py --model "T_alibi" --train_task "copy" --eval_task  "copy" --min_train_len 50 --max_train_len 300 --min_eval_len 100 --max_eval_len 1000 --steps 2000 --heads 1 --n_gram 0 --context_len 2001
# python3 synthetic_exps/main.py --model "mambapp" --train_task "copy" --eval_task  "copy" --min_train_len 50 --max_train_len 300 --min_eval_len 950 --max_eval_len 1000 --steps 20 --heads 0 --n_gram 0 --context_len 2001

# python3 synthetic_exps/main.py --model "T_rope" --train_task "copy" --eval_task  "copy" --min_train_len 5 --max_train_len 50 --min_eval_len 15 --max_eval_len 100 --steps 2000 --heads 1 --n_gram 0 --context_len 201
python3 synthetic_exps/main.py --model "mamba" --train_task "copy" --eval_task  "copy" --min_train_len 5 --max_train_len 50 --min_eval_len 15 --max_eval_len 100 --steps 20 --heads 0 --n_gram 0 --context_len 201

# T_alibi
# T_rope