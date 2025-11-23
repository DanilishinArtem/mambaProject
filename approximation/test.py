from datasets import load_dataset
from transformers import AutoTokenizer

path = ''
dataset = load_dataset("hazyresearch/based-fda", cache_dir="~/.cache/huggingface/datasets/")
print(dataset)
texts = dataset["validation"]["text"][:2]
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

encodings = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)
input_ids = encodings["input_ids"].cuda()
for idx, item in enumerate(input_ids):
    print(f'[{idx}] item: {item}')