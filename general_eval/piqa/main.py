from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# PIQA Accuracy: 0.6210 - a little bit better then random choice
# 50% — random choice
# 60–70% — weak/average result, little bit better then random choice
# 70–80% — not bad result, model really learned on the dataset
# 80–90%+ — strong model
def test():
    # Загружаем датасет PIQA
    dataset = load_dataset("piqa")

    # Загружаем твою предобученную модель
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    def evaluate(model, tokenizer, dataset, num_samples=500):
        correct = 0
        total = 0
        device = next(model.parameters()).device

        for example in dataset.select(range(num_samples)):
            premise = example["goal"]
            option1 = example["sol1"]
            option2 = example["sol2"]

            # Считаем log-likelihood для каждого варианта
            def score_option(option):
                text = premise + " " + option
                inputs = tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                return -outputs.loss.item()  # чем выше, тем лучше

            score1 = score_option(option1)
            score2 = score_option(option2)

            pred = 0 if score1 > score2 else 1
            if pred == example["label"]:
                correct += 1
            total += 1

        return correct / total

    acc = evaluate(model, tokenizer, dataset["validation"], num_samples=1000)
    print(f"PIQA Accuracy: {acc:.4f}")


if __name__ == "__main__":
    test()