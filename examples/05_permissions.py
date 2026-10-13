"""05 · Permissions: a capability the user must grant before a tool runs."""

from _common import default_model
from memoflow_agent import Agent, CapabilityPolicy, GrantDuration, InMemoryGrantStore, Principal, Resume, tool

CALENDAR = {"2026-10-14": ["09:00 Stand-up", "14:00 Design review"]}


@tool(permission="calendar.read")
def read_calendar(day: str) -> list[str]:
    """Read the user's calendar for a day (YYYY-MM-DD)."""
    return CALENDAR.get(day, [])


def main(model=None, *, grant: bool = True):
    grants = InMemoryGrantStore()  # use PostgresGrantStore for durable grants
    policy = CapabilityPolicy(grants=grants, grantable=frozenset({"calendar.read"}))
    agent = Agent(
        model=default_model(model), tools=[read_calendar], policy=policy, grants=grants, principal=Principal("user-123")
    )

    result = agent.run_sync("What is on my calendar on 2026-10-14?")
    if result.suspended and result.pending["type"] == "authorization.required":
        print("Agent asks for:", result.pending["resource"])
        # A real app shows this to the user. The model can never grant it itself.
        result = agent.resume_sync(result.run_id, Resume.authorize(grant, duration=GrantDuration.SESSION))
    print(result.output)
    return result


if __name__ == "__main__":
    main()
