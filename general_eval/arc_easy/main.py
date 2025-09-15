import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm

# small models: 25–30%, average models: 45–55%, GPT3: about 70%, GPT-4, Claude 3, Gemini: 85%+
# result 0.4263 - normal reault for average model ...

def test():
    # === 1. Загружаем датасет ARC-Easy ===
    dataset = load_dataset("ai2_arc", "ARC-Easy")

    # === 2. Загружаем твою модель и токенайзер ===
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda().eval()

    # === 3. Функция для подсчета лог-вероятности каждого варианта ===
    def score_choice(question, choice):
        prompt = f"Вопрос: {question}\nОтвет: {choice}\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            # считаем лог-вероятности токенов "choice"
            logits = outputs.logits[:, :-1, :]
            labels = inputs.input_ids[:, 1:]
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_logprobs = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            return token_logprobs.sum().item()

    # === 4. Оценка на валидации ARC-Easy ===
    correct = 0
    total = 0

    for ex in tqdm(dataset["validation"]):
        question = ex["question"]
        choices = ex["choices"]["text"]
        answer = ex["answerKey"]

        # посчитаем score для каждого варианта
        scores = [score_choice(question, c) for c in choices]
        pred_idx = int(np.argmax(scores))
        pred = ex["choices"]["label"][pred_idx]

        if pred == answer:
            correct += 1
        total += 1

    acc = correct / total
    print(f"ARC-Easy Accuracy: {acc:.4f}")


if __name__ == "__main__":
    test()