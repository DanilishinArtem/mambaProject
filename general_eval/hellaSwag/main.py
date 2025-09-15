from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# HellaSwag Accuracy: 0.2844: this is like about random choice (GPT2: 27-30%, GPT-Neo, GPT-J, LLaMA-7B: 50-70%, GPT-3, LLaMA-2-70B: 80-90%)
def test():
    # Загружаем датасет
    dataset = load_dataset("hellaswag", split="validation")

    # Загружаем модель
    model_name = "gpt2"   # замени на свою модель
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    device = next(model.parameters()).device

    correct = 0

    for item in tqdm(dataset):
        context = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])

        scores = []
        for choice in endings:
            prompt = context + " " + choice
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # считаем loss как отрицательное log-likelihood
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
            scores.append(-loss)

        pred = int(torch.tensor(scores).argmax())
        if pred == label:
            correct += 1

    accuracy = correct / len(dataset)
    print(f"HellaSwag Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    test()