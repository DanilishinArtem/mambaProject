import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ARC-Challenge Accuracy: 0.2218 this is random choice (one of the four answers)
def test():
    # загружаем ARC-Challenge
    dataset = load_dataset("ai2_arc", "ARC-Challenge")

    # модель (замени на свою)
    model_name = "gpt2"   # например
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    model.eval()

    def predict(question, choices):
        # Формируем prompt в стиле multiple-choice
        prompt = f"Вопрос: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{i+1}) {choice}\n"
        prompt += "Ответ:"

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # ищем совпадение с вариантами
        answer = answer.strip()
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return int(answer) - 1
        else:
            # fallback: берём по вероятности логитов
            logits = model(**inputs).logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            choice_scores = []
            for choice in choices:
                choice_ids = tokenizer(" " + choice, return_tensors="pt").input_ids.to("cuda")
                score = probs[0, choice_ids[0, 0]].item()
                choice_scores.append(score)
            return int(torch.tensor(choice_scores).argmax())

    # считаем Accuracy
    correct = 0
    total = 0

    for sample in dataset["test"]:
        q = sample["question"]
        choices = sample["choices"]["text"]
        label = sample["answerKey"]

        pred = predict(q, choices)
        gold = sample["choices"]["label"].index(label)

        if pred == gold:
            correct += 1
        total += 1

    print(f"ARC-Challenge Accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    test()