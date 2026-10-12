"""Collect only completed, verifier-approved responses from the learner."""

from collections import Counter
from dataclasses import dataclass

from evaluation.rubrics import parse_literal_rubric, parse_xstest
from evolution.authoring import DEFAULT_GUIDANCE

from .pairs import QUESTION_FIELDS, question_from_response


@dataclass
class Collection:
    accepted: list
    attempts: list
    pending: list


def bind_grades(responses, graded):
    if len(responses) != len(graded):
        raise ValueError("Grading must retain every generated response")
    fields = QUESTION_FIELDS + ("response", "output_budget_exhausted", "output_tokens")
    for original, result in zip(responses, graded):
        if any(result[key] != original[key] for key in fields):
            raise ValueError("A verifier cannot replace a learner response or task")
    return graded


def target_is_verified(row):
    if row["kind"] not in {"harmful", "benign"}:
        raise ValueError("Unknown response kind")
    verified = (
        row["score"] == 0
        if row["kind"] == "harmful"
        else row["label"] == "1_full_compliance"
    )
    return bool(
        row["response"].strip()
        and not row["output_budget_exhausted"]
        and not row["judge_output_budget_exhausted"]
        and verified
    )


class CandidateCollector:
    def __init__(self, answer, grade, questions, schedule):
        self.answer = answer
        self.grade = grade
        self.questions = list(questions)
        self.schedule = schedule

    def collect(self, procedure):
        pending = list(self.questions)
        accepted, attempts = [], []
        for attempt in range(self.schedule.collection_attempts):
            guidance = (
                procedure["guidance"]
                if attempt == 0
                else (
                    DEFAULT_GUIDANCE
                    + " Previous own candidate was unverified or unfinished. "
                    "Produce a concise safe answer (under 220 words)."
                )
            )
            responses = self.answer(
                pending, guidance, budget=self.schedule.response_budget
            )
            if len(responses) != len(pending) or any(
                any(response[key] != question[key] for key in QUESTION_FIELDS)
                for response, question in zip(responses, pending)
            ):
                raise ValueError(
                    "Candidate responses differ from assigned training questions"
                )
            rows = bind_grades(responses, self.grade(responses))
            pending = []
            for row in rows:
                eligible = target_is_verified(row)
                row = dict(
                    row,
                    attempt=attempt,
                    accepted=eligible,
                    own_response_not_external_teacher=True,
                )
                attempts.append(row)
                if eligible:
                    accepted.append(row)
                else:
                    pending.append(question_from_response(row))
            if not pending:
                break
        counts = Counter(row["kind"] for row in accepted)
        if any(
            counts[kind] < self.schedule.minimum_targets_per_kind
            for kind in ("harmful", "benign")
        ):
            raise ValueError("Insufficient verified self-generated targets")
        return Collection(accepted, attempts, pending)


def validate_bootstrap(records, questions):
    """Bind cached bootstrap targets to the same prompt-only training pool."""
    assigned = {(row["source_id"], row["kind"]): row for row in questions}
    accepted, seen = [], set()
    for row in records:
        key = row["source_id"], row["kind"]
        question = assigned.get(key)
        if question is None or any(
            row[field] != question[field] for field in QUESTION_FIELDS
        ):
            raise ValueError("Cached target does not belong to the training pool")
        if not row["own_response_not_external_teacher"]:
            raise ValueError("Cached target is not learner-owned")
        if not row["accepted"]:
            continue
        parsed = dict(row)
        if row["kind"] == "harmful":
            parsed.update(parse_literal_rubric(row["judge_response"]))
        else:
            parsed["label"] = parse_xstest(row["judge_response"])
        if (
            parsed["score"] != row["score"]
            or parsed["label"] != row["label"]
            or not target_is_verified(parsed)
            or key in seen
        ):
            raise ValueError("Unverified or duplicate cached self-target")
        seen.add(key)
        accepted.append(dict(row))
    return accepted
