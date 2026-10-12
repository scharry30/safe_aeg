"""Bounded, chronological retention of verified learner responses."""


class ReplayBuffer:
    def __init__(self, initial, capacity=240):
        if capacity <= 0:
            raise ValueError("Positive replay capacity required")
        self.capacity = capacity
        self.rows = list(initial)

    def extend(self, rows):
        # Repeated tasks from different generations remain distinct targets.
        self.rows = (self.rows + list(rows))[-self.capacity :]
        return self.rows
