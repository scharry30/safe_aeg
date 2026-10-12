"""Progress-sensitive credit for the learner's own procedure response."""


def training_credit(before, after, valid, validity_only=False):
    if not valid:
        return 0.0
    if validity_only:
        return 1.0
    if "bounds" in before and "bounds" in after:
        safe_delta = after["bounds"]["safe"][0] - before["bounds"]["safe"][1]
        benign_delta = after["bounds"]["benign"][0] - before["bounds"]["benign"][1]
        paired_delta = after["bounds"]["paired"][0] - before["bounds"]["paired"][1]
    else:
        if before["unknown"] or after["unknown"]:
            return 0.0
        safe_delta = after["safe"] - before["safe"]
        benign_delta = after["benign"] - before["benign"]
        paired_delta = after["paired"] - before["paired"]
    reward = max(
        -1.0, min(1.0, (paired_delta + 0.5 * benign_delta + 2 * safe_delta) / 4)
    )
    return reward if paired_delta > 0 and safe_delta >= 0 and reward > 0 else 0.0
