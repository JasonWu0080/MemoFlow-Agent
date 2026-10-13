"""08 · A small knowledge agent: search, then read, then cite.

The documents are private, so reading them needs the "docs.read"
capability (granted here by the principal's role). Tool results are
marked as untrusted data, so instructions hidden in a document cannot
unlock actions.
"""

from _common import default_model
from memoflow_agent import Agent, CapabilityPolicy, Principal, ToolError, ToolResult, tool

DOCS = {
    "d1": ("Refund policy", "Refunds are possible within 30 days of purchase with a receipt."),
    "d2": ("Shipping", "Orders ship within 2 business days. Express shipping costs 9 EUR."),
}


@tool(permission="docs.read", public_fields=("query",))
def search_docs(query: str, limit: int = 3) -> ToolResult:
    """Search the knowledge base. Returns ids, titles and short snippets."""
    terms = query.lower().split()
    hits = [
        {"id": doc_id, "title": title, "snippet": body[:80]}
        for doc_id, (title, body) in DOCS.items()
        if any(term in (title + " " + body).lower() for term in terms)
    ][:limit]
    return ToolResult(observation={"results": hits}, summary={"count": len(hits)}, untrusted_texts=[h["snippet"] for h in hits])


@tool(permission="docs.read")
def read_doc(doc_id: str) -> ToolResult:
    """Read one document by id. Cite it as [doc_id]."""
    if doc_id not in DOCS:
        raise ToolError("not_found", "No such document.")
    title, body = DOCS[doc_id]
    return ToolResult(observation={"id": doc_id, "title": title, "text": body}, untrusted_texts=[body])


def main(model=None):
    agent = Agent(
        model=default_model(model),
        tools=[search_docs, read_doc],
        instructions="Answer from the knowledge base only. Cite documents as [doc_id].",
        policy=CapabilityPolicy(),
        principal=Principal("support-agent", capabilities=frozenset({"docs.read"})),
    )
    result = agent.run_sync("Can I get a refund after three weeks?")
    print(result.output)
    return result


if __name__ == "__main__":
    main()
