import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# result 0.5038 - random choice because we choose one of the two answers ...
def test():
    # если у тебя свой model + tokenizer, подставь сюда
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda().eval()

    dataset = load_dataset("winogrande", "winogrande_xl")["validation"]

    def score_sentence(sentence):
        """Считает loglikelihood текста"""
        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return -loss.item()

    correct = 0
    for example in tqdm(dataset):
        sent = example["sentence"]
        opt1 = sent.replace("_", example["option1"])
        opt2 = sent.replace("_", example["option2"])
        score1 = score_sentence(opt1)
        score2 = score_sentence(opt2)
        pred = "1" if score1 > score2 else "2"
        if pred == example["answer"]:
            correct += 1

    acc = correct / len(dataset)
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    test()