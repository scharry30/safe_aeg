"""Carry forward only training-pool failures, not held-out outcomes."""


def failure_feedback(training_metrics):
    return {
        "failed_categories": dict(training_metrics["failed_categories"]),
        "safe": training_metrics["safe"],
        "benign": training_metrics["benign"],
        "paired": training_metrics["paired"],
    }
