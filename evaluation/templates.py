"""Read literal official templates without executing their source files."""

import ast


def harmbench_templates(source):
    assignments = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "LLAMA2_CLS_PROMPT"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise ValueError("A unique classifier template assignment is required")
    templates = ast.literal_eval(assignments[0].value)
    if not isinstance(templates, dict) or not all(
        isinstance(templates.get(key), str) for key in ("prompt", "prompt_contextual")
    ):
        raise ValueError("Invalid classifier template dictionary")
    return templates


def xstest_prompt(question, response, source):
    assignments = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "PROMPT"
            for target in node.targets
        )
    ]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.JoinedStr):
        raise ValueError("A unique official XSTest f-string template is required")
    parts = []
    for part in assignments[0].value.values:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            parts.append(part.value)
        elif (
            isinstance(part, ast.FormattedValue)
            and isinstance(part.value, ast.Name)
            and part.value.id in {"QUESTION", "RESPONSE"}
            and part.conversion == -1
            and part.format_spec is None
        ):
            parts.append(question if part.value.id == "QUESTION" else response)
        else:
            raise ValueError("Unsupported template expression")
    return "".join(parts)
