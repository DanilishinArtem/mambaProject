lm_eval --model hf \
    --model_args pretrained=facebook/opt-125m \
    --tasks fda \
    --device cuda \
    --batch_size 8