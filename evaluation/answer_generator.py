from vllm import SamplingParams
from tqdm import tqdm


class AnswerGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        dataset,
        instruction,
        context_lengths,
        max_context_length,
        max_new_tokens,
        chat_model,
        sys_prompt,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.instruction = instruction
        self.context_lengths = context_lengths
        self.max_context_length = max_context_length
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.chat_model = chat_model
        self.sys_prompt = sys_prompt

    def create_prompt(self, sample):
        prompt = self.instruction
        prompt = prompt.replace("{context}", sample["context"])
        prompt = prompt.replace("{input}", sample["input"])
        return prompt

    def create_prompt_with_chat_template(self, sample):
        prompt = self.instruction
        prompt = prompt.replace("{context}", sample["context"])
        prompt = prompt.replace("{input}", sample["input"])
        messages = []
        if self.sys_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": self.sys_prompt,
                }
            )
        messages.append({"role": "user", "content": prompt})
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=True,
            max_length=self.max_context_length,
        )
        return prompt

    def generate_answers(self):
        generated_answers = []

        for sample in tqdm(self.dataset):
            if sample["length"] not in self.context_lengths:
                continue

            if self.chat_model:
                prompt = self.create_prompt_with_chat_template(sample)
            else:
                prompt = self.create_prompt(sample)

            inputs = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_context_length,
                return_tensors="pt",
            ).to(self.device)

            generation_output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id
            )

            prompt_len = len(inputs["input_ids"][0])

            model_answer = self.tokenizer.decode(
                generation_output[0][prompt_len:].cpu(), skip_special_tokens=True
            )
            generated_answer = {
                "length": sample["length"],
                "model_answer": model_answer,
                "positive_outputs": sample["positive_outputs"],
                "negative_outputs": sample["negative_outputs"],
            }
            generated_answers.append(generated_answer)
        return generated_answers


class vLLM_AnswerGenerator(AnswerGenerator):
    def create_prompt(self, sample):
        prompt = self.instruction
        prompt = prompt.replace("{context}", sample["context"])
        prompt = prompt.replace("{input}", sample["input"])
        sample["prompt"] = prompt
        return sample

    def create_prompt_with_chat_template(self, sample):
        prompt = self.instruction
        prompt = prompt.replace("{context}", sample["context"])
        prompt = prompt.replace("{input}", sample["input"])
        messages = []
        if self.sys_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": self.sys_prompt,
                }
            )
        messages.append({"role": "user", "content": prompt})
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            truncation=True,
            max_length=self.max_context_length,
            tokenize=False,
        )
        sample["prompt"] = text
        return sample

    def generate_answers(self):
        self.dataset = self.dataset.filter(
            lambda x: x["length"] in self.context_lengths
        )
        if self.chat_model:
            self.dataset = self.dataset.map(self.create_prompt_with_chat_template)
        else:
            self.dataset = self.dataset.map(self.create_prompt)

        generated_answers = []

        if len(self.dataset) == 0:
            return generated_answers

        out = self.model.generate(
            self.dataset["prompt"],
            sampling_params=SamplingParams(
                temperature=0.0,
                max_tokens=self.max_new_tokens,
                use_beam_search=False,
                truncate_prompt_tokens=self.max_context_length,
            ),
        )

        for ind, request in enumerate(out):
            model_answer = request.outputs[0].text
            generated_answer = {
                "length": self.dataset[ind]["length"],
                "model_answer": model_answer,
                "positive_outputs": self.dataset[ind]["positive_outputs"],
                "negative_outputs": self.dataset[ind]["negative_outputs"],
            }
            generated_answers.append(generated_answer)
        return generated_answers
