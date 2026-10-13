"""06 · Human-in-the-loop: approve one exact action before it runs."""

from _common import default_model
from memoflow_agent import Agent, Principal, Resume, tool

OUTBOX: list[dict] = []


@tool(side_effect="non_idempotent", requires_approval=True, risk="high")
def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email."""
    OUTBOX.append({"to": to, "subject": subject, "body": body})
    return {"sent": True}


def main(model=None, *, approve: bool = True):
    agent = Agent(model=default_model(model), tools=[send_email], principal=Principal("user-123"))
    result = agent.run_sync("Email ana@example.com that the report is ready.")
    while result.suspended and result.pending["type"] == "approval.required":
        print("Approve", result.pending["tool"], result.pending["arguments"], "?")
        # The approval is bound to these exact arguments; it cannot be reused.
        decision = Resume.approve(result.pending["approval_id"]) if approve else Resume.reject(result.pending["approval_id"])
        result = agent.resume_sync(result.run_id, decision)
    print(result.output)
    return result


if __name__ == "__main__":
    main()
