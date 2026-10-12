"""Persistent behavior learning followed by credited procedure consolidation."""

import random

from adaptation.lora import inject_lora
from adaptation.objective import ResponseObjective
from adaptation.optimizer import ResponseOptimizer
from adaptation.parameters import (
    adapter_state,
    parameter_fingerprint,
    restore_adapter,
    seed_parameters,
)
from data.pairs import pair_questions, validate_questions
from data.replay import ReplayBuffer
from data.targets import CandidateCollector, bind_grades
from evaluation.paired import paired_metrics
from inference.learner import LearnerInference
from inference.verifier import FrozenVerifier
from protocol.schedule import TrainingSchedule
from protocol.variants import MAIN_METHODS

from .authoring import DEFAULT_GUIDANCE, author_procedure
from .credit import training_credit
from .feedback import failure_feedback
from .state import EvolutionState


class SafetyEvolution:
    def __init__(
        self,
        student,
        tokenizer,
        judge,
        judge_tokenizer,
        training_roots,
        development_roots,
        templates,
        xstest_source,
        schedule=None,
        swap_verifier=False,
    ):
        if student is judge:
            raise ValueError("Learner and verifier must be separate models")
        training_roots = list(training_roots)
        development_roots = list(development_roots)
        self.schedule = schedule or TrainingSchedule()
        self.student = student
        self.tokenizer = tokenizer
        for roots in (training_roots, development_roots):
            if any(
                row["externally_supplied_response_target"] is not None for row in roots
            ):
                raise ValueError("Only prompt-only task roots are allowed")
        self.training = validate_questions(pair_questions(training_roots))
        self.development = validate_questions(pair_questions(development_roots))
        if {row["source_id"] for row in self.training} & {
            row["source_id"] for row in self.development
        }:
            raise ValueError("Training and development roots must be disjoint")
        self.categories = sorted({row["category"] for row in self.training})
        self.fixed = dict(focus_category=self.categories[0], guidance=DEFAULT_GUIDANCE)
        seed_parameters(self.schedule.seed)
        inject_lora(student, self.schedule.lora_rank, self.schedule.lora_alpha)
        self.initial = adapter_state(student)
        self.initial_hash = parameter_fingerprint(self.initial)
        self.rng = random.Random(self.schedule.seed)
        self.answer = LearnerInference(
            student, tokenizer, self.schedule.learner_batch_size
        )
        self.grade = FrozenVerifier(
            judge,
            judge_tokenizer,
            templates,
            xstest_source,
            student,
            self.schedule,
            swap=swap_verifier,
        )
        self.collector = CandidateCollector(
            self.answer, self.grade, self.training, self.schedule
        )
        self.objective = ResponseObjective(student, tokenizer)
        self.optimizer = ResponseOptimizer(
            student, self.objective, self.rng, self.schedule
        )

    def restore_initial(self):
        restore_adapter(self.student, self.initial)
        self.rng = random.Random(self.schedule.seed)
        self.optimizer.rng = self.rng
        self.optimizer.credited_writer_steps = 0

    def probe(self, questions):
        responses = self.answer(questions, budget=self.schedule.response_budget)
        return paired_metrics(bind_grades(responses, self.grade(responses)))

    def evolve(self, method, bootstrap, on_generation=None):
        state = EvolutionState(
            procedure=dict(self.fixed),
            replay=ReplayBuffer(bootstrap, self.schedule.replay_capacity),
        )
        learning_rate = (
            method.learning_rate
            if method.learning_rate is not None
            else self.schedule.learning_rate
        )
        for generation in range(1, self.schedule.generations + 1):
            parent_hash = parameter_fingerprint(adapter_state(self.student))
            before = self.probe(self.training)
            if not method.inherit_procedure:
                state.reset_context(self.fixed)
            authored, writer = None, None
            if method.adaptive_procedure:
                authored, writer = author_procedure(
                    self.answer,
                    self.categories,
                    state.procedure,
                    state.feedback,
                    budget=self.schedule.writer_budget,
                )
                if authored is not None:
                    state.procedure = authored
            collection = (
                None
                if method.reuse_bootstrap
                else self.collector.collect(state.procedure)
            )
            corpus = bootstrap if method.reuse_bootstrap else collection.accepted
            behavior = self.optimizer.optimize(
                corpus,
                self.schedule.behavior_steps,
                learning_rate,
                focus=state.procedure["focus_category"]
                if method.adaptive_procedure
                else None,
            )
            intermediate = self.probe(self.training)
            credit = (
                training_credit(
                    before,
                    intermediate,
                    authored is not None,
                    method.validity_only_credit,
                )
                if method.writer_supervision
                else 0.0
            )
            replay = state.replay.extend(corpus)
            consolidation = self.optimizer.optimize(
                bootstrap if method.reuse_bootstrap else replay,
                self.schedule.consolidation_steps,
                learning_rate,
                writer=writer if method.writer_supervision else None,
                credit=credit,
            )
            # Development is reporting-only and never enters authoring or credit.
            development = self.probe(self.development)
            record = dict(
                method=method.name,
                generation=generation,
                procedure=dict(state.procedure),
                valid_new_author=authored is not None,
                writer_credit=credit,
                parent_adapter_hash=parent_hash,
                adapter_hash=parameter_fingerprint(adapter_state(self.student)),
                training_before=before,
                training_intermediate=intermediate,
                development=development,
                behavior=behavior,
                consolidation=consolidation,
                accepted_targets=len(corpus),
                candidate_attempts=0
                if collection is None
                else len(collection.attempts),
                retained_even_if_regressed=True,
            )
            state.retain(record, failure_feedback(intermediate))
            if on_generation is not None:
                on_generation(state, self.student, self.rng)
        return state

    def run(self, methods=MAIN_METHODS, on_generation=None):
        names = [method.name for method in methods]
        if not names or len(names) != len(set(names)):
            raise ValueError("Nonempty, unique method assignments required")
        self.restore_initial()
        bootstrap = self.collector.collect(self.fixed)
        states = {}
        for method in methods:
            self.restore_initial()
            states[method.name] = self.evolve(method, bootstrap.accepted, on_generation)
        return dict(
            initial_adapter_hash=self.initial_hash, bootstrap=bootstrap, methods=states
        )
