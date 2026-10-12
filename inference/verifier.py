"""Frozen response grading; verifier text never becomes a training target."""

from evaluation.rubrics import parse_literal_rubric, parse_xstest
from evaluation.templates import xstest_prompt

from .decoding import generate_greedy
from .placement import verifier_device


class FrozenVerifier:
    def __init__(
        self, model, tokenizer, templates, xstest_source, student, schedule, swap=False
    ):
        if model is student:
            raise ValueError("Learner and frozen verifier must be separate models")
        self.model = model
        self.tokenizer = tokenizer
        self.templates = templates
        self.xstest_source = xstest_source
        self.student = student
        self.schedule = schedule
        self.swap = swap
        self.device = next(student.parameters()).device
        self.generated_tokens = 0
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()

    def __call__(self, rows):
        prompts = []
        for row in rows:
            if row["kind"] == "harmful":
                messages = [
                    dict(
                        role="system",
                        content=self.templates["strongreject_rubric_system"],
                    ),
                    dict(
                        role="user",
                        content=self.templates["strongreject_rubric"].format(
                            forbidden_prompt=row["prompt"],
                            response=row["response"],
                        ),
                    ),
                ]
            elif row["kind"] == "benign":
                messages = [
                    dict(role="system", content="You are a helpful assistant."),
                    dict(
                        role="user",
                        content=xstest_prompt(
                            row["prompt"],
                            row["response"],
                            self.xstest_source,
                        ),
                    ),
                ]
            else:
                raise ValueError("Only paired response kinds may be graded")
            prompts.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        with verifier_device(self.student, self.model, self.device, self.swap):
            values = generate_greedy(
                self.model,
                self.tokenizer,
                prompts,
                self.schedule.judge_budget,
                batch_size=self.schedule.judge_batch_size,
                device=self.device,
            )
        self.generated_tokens += sum(value["output_tokens"] for value in values)
        graded = []
        for row, value in zip(rows, values):
            result = dict(
                row,
                judge_response=value["text"],
                judge_output_budget_exhausted=value["output_budget_exhausted"],
            )
            if row["kind"] == "harmful":
                result.update(parse_literal_rubric(value["text"]), label=None)
            else:
                result.update(score=None, label=parse_xstest(value["text"]))
            graded.append(result)
        return graded
