okay piunt"""
Module 8: Citation Formatting

The LLM already inlines [Page X] citations (per Module 7's prompt), but
LLMs aren't 100% reliable about following instructions. This module builds
an explicit, guaranteed-accurate "Sources" list from the retrieved chunk
metadata itself — not from anything the LLM generated — and appends it to
the answer so citations are always trustworthy even if the model forgets.
"""


def format_sources(retrieved_chunks: list) -> str:
    """
    Builds a deduplicated, human-readable source list from retrieved chunks.
    Example output:

    Sources:
    - Page 2 (text)
    - Page 5 (table, near 'Q3 Financial Summary')
    """
    if not retrieved_chunks:
        return "Sources: none"

    seen = set()
    lines = []
    for chunk in retrieved_chunks:
        meta = chunk["metadata"]
        page = meta["page_num"]
        ctype = meta.get("chunk_type", "prose")
        heading = meta.get("heading", "")

        key = (page, ctype, heading)
        if key in seen:
            continue
        seen.add(key)

        if ctype == "table" and heading:
            lines.append(f"- Page {page} (table, near '{heading}')")
        else:
            lines.append(f"- Page {page} ({ctype})")

    return "Sources:\n" + "\n".join(sorted(lines))


def format_answer_with_sources(answer: str, retrieved_chunks: list) -> str:
    """Combines the LLM's answer with the guaranteed-accurate source list."""
    return f"{answer.strip()}\n\n{format_sources(retrieved_chunks)}"


if __name__ == "__main__":
    fake_chunks = [
        {"text": "...", "metadata": {"page_num": 2, "chunk_type": "prose"}},
        {"text": "...", "metadata": {"page_num": 5, "chunk_type": "table", "heading": "Q3 Summary"}},
    ]
    print(format_answer_with_sources("Revenue was $4.2M.", fake_chunks))
