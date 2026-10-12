"""The same learner answers tasks and authors its next procedure."""

from .decoding import generate_greedy


class LearnerInference:
    def __init__(self, model, tokenizer, batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.generated_tokens = 0

    def __call__(self, questions, guidance=None, budget=512):
        prompts = []
        for question in questions:
            messages = [dict(role="system", content=guidance)] if guidance else []
            messages += [dict(role="user", content=question["prompt"])]
            prompts.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        values = generate_greedy(
            self.model,
            self.tokenizer,
            prompts,
            budget,
            batch_size=self.batch_size,
        )
        self.generated_tokens += sum(value["output_tokens"] for value in values)
        return [
            dict(
                question,
                response=value["text"],
                output_tokens=value["output_tokens"],
                output_budget_exhausted=value["output_budget_exhausted"],
            )
            for question, value in zip(questions, values)
        ]
