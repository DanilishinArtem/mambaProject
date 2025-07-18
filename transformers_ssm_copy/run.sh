python3 synthetic_exps/main.py --model "mambapp" --train_task "copy" --eval_task  "copy" --min_train_len 5 --max_train_len 20 --min_eval_len 15 --max_eval_len 20 --steps 2000 --heads 4

# одну мамбу в качестве всех голов
# поиграться с d_state
# сравнить эффект повышения d_state и эффект повышения голов 