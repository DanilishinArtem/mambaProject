# rm -rf /home/adanilishin/mambaProject/transformers_ssm_copy/tensorboard
# python3 synthetic_exps/main.py --model "mambapp" --train_task "suffix_ngram" --eval_task  "suffix_ngram" --min_train_len 150 --max_train_len 200 --min_eval_len 195 --max_eval_len 200 --steps 500 --heads 1 --n_gram 100 --context_len 420

# rm -rf /home/adanilishin/mambaProject/transformers_ssm_copy/tensorboard/model_mambapp_layer_12_hidden_1024_heads_8_train_copy_lr_1e-05_epochs_1_steps_300
python3 synthetic_exps/main.py --model "mambapp" --train_task "copy" --eval_task  "copy" --min_train_len 50 --max_train_len 100 --min_eval_len 95 --max_eval_len 100 --steps 2000 --heads 1 --n_gram 0 --context_len 220 #--layers 1

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