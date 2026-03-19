def _to_str(content) -> str:
    if isinstance(content, str):
        return content

    return " ".join(c if isinstance(c, str) else str(c) for c in content)
