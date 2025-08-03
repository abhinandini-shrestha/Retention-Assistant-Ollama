def format_description_md(text: str) -> str:
    lines = text.splitlines()
    formatted_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("•"):
            # Convert bullet to markdown list
            formatted_lines.append(f"- {stripped[1:].strip()}")
        else:
            formatted_lines.append(stripped)
    return "\n".join(formatted_lines)