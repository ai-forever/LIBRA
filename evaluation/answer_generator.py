import os
import json

from tqdm import tqdm


class AnswerGenerator():
    def __init__(self, model, tokenizer, device, dataset, instruction, context_lengths, max_context_length, max_new_tokens):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.instruction = instruction
        self.context_lengths = context_lengths
        self.max_context_length = max_context_length
        self.device = device
        self.max_new_tokens = max_new_tokens

    def create_prompt(self, sample):
        prompt = self.instruction
        prompt = prompt.replace("{context}", sample["context"])
        prompt = prompt.replace("{input}", sample["input"])
        return prompt      
        
    def generate_answers(self):
        generated_answers = []
        for sample in tqdm(self.dataset):
            if sample["length"] not in self.context_lengths: continue
            prompt = self.create_prompt(sample)
            inputs = self.tokenizer(prompt, truncation=True, max_length=self.max_context_length, return_tensors="pt").to(self.device)
            generation_output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0
                )
            model_answer = self.tokenizer.decode(generation_output[0][-self.max_new_tokens:].cpu())
            generated_answer = {
                "length": sample["length"],
                "model_answer": model_answer,
                "positive_outputs": sample["positive_outputs"],
                "negative_outputs": sample["negative_outputs"]
            }
            generated_answers.append(generated_answer)
        return generated_answers
