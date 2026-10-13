"""Web search authorization: may this model turn search, and how?

Decided by the runtime before every model turn — never by the model.

* ``OFF`` — no search tool of any kind is declared.
* ``HOSTED`` — the provider's own search tool is declared in the request.
  Only allowed when the request holds **no private context**: whatever is in
  a request with a hosted search tool may end up in a query the provider
  writes, and the application cannot inspect or veto it.
* ``ISOLATED`` — the request holds private context. The hosted tool is
  withheld; an application tool performs the search in a *separate* request
  containing only a query that passed :func:`check_isolated_query`.

There is never an automatic fallback to another provider's search.
The privacy check is mechanical and documented as a heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .safety import PrivacyVerdict, check_query_privacy


class SearchMode(StrEnum):
    OFF = "off"
    HOSTED = "hosted"
    ISOLATED = "isolated"


@dataclass(frozen=True)
class SearchDecision:
    mode: SearchMode
    reason: str | None = None

    @property
    def offered(self) -> bool:
        return self.mode is not SearchMode.OFF


def decide_search(
    *,
    enabled: bool,
    model_supports_search: bool,
    tools_enabled: bool = True,
    opted_out: bool = False,
    user_forbade: bool = False,
    run_limit_reached: bool = False,
    unavailable_reason: str | None = None,
    private_context: bool = False,
    personal_data: bool = False,
) -> SearchDecision:
    """The decision order MemoFlow's Web Search V5 uses, without its data sources."""
    if not enabled:
        return SearchDecision(SearchMode.OFF, "disabled")
    if opted_out:
        return SearchDecision(SearchMode.OFF, "opted_out")
    if user_forbade:
        return SearchDecision(SearchMode.OFF, "user_forbade")
    if not tools_enabled or not model_supports_search:
        return SearchDecision(SearchMode.OFF, "unsupported")
    if run_limit_reached:
        return SearchDecision(SearchMode.OFF, "run_limit")
    if unavailable_reason is not None:
        return SearchDecision(SearchMode.OFF, unavailable_reason)
    if private_context:
        return SearchDecision(SearchMode.ISOLATED, "private_context")
    if personal_data:
        return SearchDecision(SearchMode.ISOLATED, "personal_data")
    return SearchDecision(SearchMode.HOSTED)


def check_isolated_query(query: str, *, private_texts: list[str], user_texts: list[str]) -> PrivacyVerdict:
    """Gate for a query leaving an ISOLATED run (see :func:`memoflow_agent.safety.check_query_privacy`)."""
    return check_query_privacy(query, private_texts=private_texts, user_texts=user_texts)
