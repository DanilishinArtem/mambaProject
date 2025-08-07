# rm -rf /home/adanilishin/mambaProject/transformers_ssm_copy/tensorboard
# python3 synthetic_exps/main.py --model "mambapp" --train_task "copy" --eval_task  "copy" --min_train_len 5 --max_train_len 100 --min_eval_len 95 --max_eval_len 100 --steps 2000 --heads 0 --n_gram 0 --context_len 220
# python3 synthetic_exps/main.py --model "mambapp" --train_task "copy" --eval_task  "copy" --min_train_len 5 --max_train_len 20 --min_eval_len 15 --max_eval_len 20 --steps 2000 --heads 0 --n_gram 0 --context_len 44

# rm -rf /home/adanilishin/mambaProject/transformers_ssm_copy/tensorboard/model_mambapp_layer_12_hidden_1024_heads_8_train_copy_lr_1e-05_epochs_1_steps_300

# одну мамбу в качестве всех голов
# поиграться с d_state
# сравнить эффект повышения d_state и эффект повышения голов 
# T_hard_alibi

# mambapp
# lstm
# T_rope
# T_nope
# T_alibi
# mamba


# Tasks:
#     copy
#     prefix_ngram
#     suffix_ngram




# python3 synthetic_exps/main.py --train_batch_size 2 --state_dim 16 --context_len 220 --model "mamba" --train_task "copy" --eval_task  "copy" --min_train_len 5 --max_train_len 100 --min_eval_len 5 --max_eval_len 100 --steps 10000 --heads 1  --epochs 1 --layers 12



rm -rf /home/adanilishin/mambaProject/transformers_ssm_copy/tensorboard/model_mambapp*
python3 synthetic_exps/main.py --train_batch_size 2 --state_dim 16 --context_len 220 --model "mambapp" --train_task "copy" --eval_task  "copy" --min_train_len 5 --max_train_len 100 --min_eval_len 5 --max_eval_len 100 --steps 10000 --heads 1  --epochs 1 --layers 12
