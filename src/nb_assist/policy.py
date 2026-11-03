"""Policy pre-filter.

Three classes of input go straight to handoff before retrieval runs:
  - other-person account data
  - real-time market data (gold, exchange rates, share prices)
  - prompt-injection attempts (instruction override)

Generic categories, not a per-question blocklist. A blocklist would
hide weak retrieval; categories are honest and generalize to new
queries. Each rule documents the trigger so the trace can report
which rule fired, not just a boolean.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Each pattern is paired with a short id used in the trace.
# Patterns are case-insensitive after normalize(); we still compile
# with IGNORECASE as a safety net for any caller that skips normalize.

_OTHER_PERSON = re.compile(
    r"\b(friend|relative|brother|sister|mother|father|wife|husband|son|daughter)"
    r"\b.{0,40}\b(account|balance|statement|deposit|transaction|info|details|"
    r"number|amount)\b",
    re.IGNORECASE,
)

_OTHER_PERSON_BN = re.compile(
    r"(বন্ধু|রিশতাদার|ভাই|বোন|মা|বাবা|স্ত্রী|স্বামী|ছেলে|মেয়ে)"
    r".{0,40}(অ্যাকাউন্ট|ব্যালেন্স|স্টেটমেন্ট|জমা|লেনদেন|তথ্য|নম্বর|টাকা)"
)

# Live market data: gold, dollar, exchange rate, share price.
_MARKET = re.compile(
    r"\b(today'?s?|current|live|rate|price|exchange|dollar|gold|silver|"
    r"share\s+price)\b",
    re.IGNORECASE,
)
_MARKET_BN = re.compile(r"আজকের|বর্তমান|দাম|রেট|অর্ডার|শেয়ার")
_MARKET_BANGLISH = re.compile(
    r"\b(ajker|current|rate|price|dollar|gold|share)\b", re.IGNORECASE
)

# Instruction override: the customer tries to change the rules.
_INJECTION = re.compile(
    r"\b(ignore|disregard|forget)\s+(the\s+)?(rules?|instructions?|policy|"
    r"guidelines?)\b|"
    r"\b(system|admin|root|developer)\s+(prompt|message|instruction)\b|"
    r"\bshow\s+me\s+(all|every|the)\s+(faq|data|customer|account)\b",
    re.IGNORECASE,
)
_INJECTION_BN = re.compile(r"নিয়ম|নির্দেশ|ভুলে|এড়িয়ে|সিস্টেম|অ্যাডমিন")


@dataclass
class PolicyDecision:
    """Decision returned by the pre-filter.

    handoff=True means skip retrieval and go straight to IDK template.
    rule is a short id ('other_person', 'market', 'injection') or None.
    matched_text is a short excerpt for the trace, never the full query.
    """
    handoff: bool
    rule: str | None = None
    matched_text: str | None = None


def _excerpt(text: str, match: re.Match) -> str:
    """Up to 60 chars around the match, for tracing only."""
    s = max(0, match.start() - 20)
    e = min(len(text), match.end() + 20)
    return text[s:e].strip()


def check_policy(text: str) -> PolicyDecision:
    """Return a policy decision for a normalized query.

    Order matters: injection first (a malicious query that also asks
    for a friend's balance is primarily an injection attempt), then
    other-person, then market. The first match wins.
    """
    if not text:
        return PolicyDecision(handoff=False)

    m = _INJECTION.search(text) or _INJECTION_BN.search(text)
    if m:
        return PolicyDecision(handoff=True, rule="injection",
                              matched_text=_excerpt(text, m))

    m = _OTHER_PERSON.search(text) or _OTHER_PERSON_BN.search(text)
    if m:
        return PolicyDecision(handoff=True, rule="other_person",
                              matched_text=_excerpt(text, m))

    m = (_MARKET.search(text) or _MARKET_BN.search(text)
         or _MARKET_BANGLISH.search(text))
    if m:
        return PolicyDecision(handoff=True, rule="market",
                              matched_text=_excerpt(text, m))

    return PolicyDecision(handoff=False)