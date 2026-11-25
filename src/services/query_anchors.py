"""Recognize the explicit identifiers and constraints a retrieval rewrite must preserve."""

from __future__ import annotations

import re
from itertools import pairwise

from utils.text import ENGLISH_STOPWORDS

_NUMBER = r"[+\u2212-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
_CURRENCY = r"[$€£]"
_IDENTIFIER = r"\w+(?:[./:-]\w+)*"
_ORDINAL = re.compile(r"\d+(?:st|nd|rd|th)", re.I)
_COMPARISON = re.compile(r"(?<!\s)\s*(?P<op><=|>=|!=|[≤≥≠<>])\s*")
_NUMERIC_PREFIX = re.compile(rf"{_CURRENCY}?[+\u2212-]?\.?\d")
_OPERATOR_ALIASES = {"≤": "<=", "≥": ">=", "≠": "!="}
_REVERSE_OPERATOR = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "!=": "!="}
_EQUALS = re.compile(r"""[\x60"']?\s*==?(?!=)\s*""")

_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
    r"|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_DAY = r"(?:0?[1-9]|[12]\d|3[01])"
_DATE = (
    rf"(?:\b(?P<month_first>{_MONTH})\.?\s+(?:the\s+)?(?P<day_after>{_DAY})(?:st|nd|rd|th)?\b"
    rf"|\b(?P<day_first>{_DAY})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<month_after>{_MONTH})\b\.?)"
    r"(?:(?:\s*,\s*|\s+)(?P<year>\d{4})\b)?"
)

# Match a rate before a standalone measure. The denominator is part of the quantity.
_UNIT = r"(?:second|minute|hour|day|week|month|year|byte|bit)s?\b|(?:[kmgt]i?)?b\b|%|percent\b"
_PER = r"(?:\s*/\s*|\s+per\s+)"
_RATE_UNIT = r"[a-z]+|%"
_DENOMINATOR = rf"{_PER}[a-z]+\b"
_PRICE = rf"(?P<price_sign>[+\u2212-]?)(?P<currency>{_CURRENCY})(?P<price>{_NUMBER})"
_QUANTITY = (
    rf"(?:{_PRICE}|(?P<rate_amount>{_NUMBER})[\s-]*(?P<numerator>{_RATE_UNIT}))"
    rf"(?P<denominators>(?:{_DENOMINATOR})+)"
    rf"|(?P<amount>{_NUMBER})[\s-]*(?P<unit>{_UNIT})"
)
_VALUE = re.compile(
    rf"{_DATE}|{_QUANTITY}|{_CURRENCY}?[+\u2212-]?(?:\.\d+|\w+)(?:[./:-]\w+|,\d{{3}}(?!\d))*%?",
    re.I,
)
_UNIT_ALIASES = {
    "s": "second",
    "sec": "second",
    "min": "minute",
    "h": "hour",
    "hr": "hour",
    "d": "day",
    "wk": "week",
    "yr": "year",
    "percent": "%",
}
_PLURALIZABLE_UNITS = {
    "second",
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "year",
    "byte",
    "bit",
    "request",
    "user",
    *_UNIT_ALIASES,
}

_KINDS = {"product", "model", "plan", "policy", "option", "section", "appendix", "region"}
_QUOTES = "\x60\"'\u201c\u201d\u2018\u2019"
_WORD = re.compile(rf"[{_QUOTES}]?{_IDENTIFIER}[{_QUOTES}]?")
# In "give the plan a longer life", lowercase a is an article. In "product a in Europe",
# the preposition ends the name. Quoting a single letter is also unambiguous.
_AFTER_A = re.compile(r"\s*(?:$|[?!.,;:)]|\b(?:in|on|at|of|for|from|with|and|or|is|was)\b)", re.I)
_QUOTED = (
    r"""\x60[^\x60\n]+\x60|"[^"\n]+"|'[^'\n]+'|\u201c[^\u201d\n]+\u201d|\u2018[^\u2019\n]+\u2019"""
)
_CASE_SENSITIVE = re.compile(r"\bcase[- ]sensitive\b", re.I)
_KEY_LABEL = re.compile(
    r"\b(?:case[- ]sensitive\s+)?(?:(?:json|object)\s+)?"
    r"(?:keys?|fields?|propert(?:y|ies)|identifiers?|names?)\b",
    re.I,
)
_KEY_ITEM = re.compile(rf"(?<!\w)(?:{_QUOTED})|{_IDENTIFIER}")
_LIST_JOIN = re.compile(r"(?:\s*,\s*(?:(?:and|or)\s+)?|\s+(?:and|or|with)\s+)", re.I)
_NAME_INTRO = re.compile(r"(?:\s*:\s*|\s+)(?:(named|called)\s+)?", re.I)
_LIST_END = re.compile(r"\s*(?:$|[.,;:?!)}\]]|\b(?:in|of|for|on|from|within|during)\b)", re.I)
_SPACE = re.compile(r"\s+")
_QUOTED_VALUE = re.compile(_QUOTED)
_QUOTED_OPERAND = re.compile(rf"(?<!\w)(?:{_QUOTED})(?!\w)")

type Anchor = tuple[str, ...]


def _unit_name(unit: str) -> str:
    if unit in {"B", "b"}:
        return "byte" if unit == "B" else "bit"
    if re.fullmatch(r"[kmgt]i?b", unit, re.I):
        # As in MB versus Mb, the last letter distinguishes bytes from bits.
        return unit[:-1].casefold() + unit[-1]
    name = unit.casefold()
    if name.removesuffix("s") in _PLURALIZABLE_UNITS:
        name = name.removesuffix("s")
    return _UNIT_ALIASES.get(name, name)


def _value(match: re.Match[str]) -> Anchor:
    month = match["month_first"] or match["month_after"]
    if month:
        day = match["day_first"] or match["day_after"]
        date = ("date", month[:3].casefold(), str(int(day)))
        return (*date, match["year"]) if match["year"] else date
    if match["denominators"]:
        amount = (match["price_sign"] or "") + (match["price"] or match["rate_amount"])
        numerator = match["currency"] or match["numerator"]
        denominators = re.split(_PER, match["denominators"], flags=re.I)[1:]
        return (
            "rate",
            amount.replace(",", ""),
            _unit_name(numerator),
            *map(_unit_name, denominators),
        )
    if match["amount"]:
        return ("quantity", match["amount"].replace(",", ""), _unit_name(match["unit"]))
    token = match.group().casefold()
    return ("token", token.replace(",", "") if re.fullmatch(_NUMBER, token) else token)


def _is_bound(value: Anchor | None) -> bool:
    return value is not None and (
        value[0] in {"date", "quantity", "rate"}
        or (value[0] == "token" and bool(_NUMERIC_PREFIX.match(value[1])))
    )


def _comparisons(text: str, starts: dict[int, Anchor], ends: dict[int, Anchor]) -> set[Anchor]:
    """Keep the same field, operator and whole bound, including in a chained comparison."""
    comparisons: set[Anchor] = set()
    for match in _COMPARISON.finditer(text):
        op = _OPERATOR_ALIASES.get(match["op"], match["op"])
        field, bound = ends.get(match.start()), starts.get(match.end())
        if _is_bound(field) and not _is_bound(bound):
            field, bound = bound, field
            op = _REVERSE_OPERATOR[op]
        if bound is None or not _is_bound(bound):
            continue
        comparisons.add(("limit", op, *bound))
        if field and (
            field[0] == "quoted-field"
            or (field[0] == "token" and field[1] not in ENGLISH_STOPWORDS)
        ):
            comparisons.add(("field-limit", field[1], op, *bound))
    return comparisons


def _constraints(text: str) -> set[Anchor]:
    anchors: set[Anchor] = set()
    starts: dict[int, Anchor] = {}
    ends: dict[int, Anchor] = {}
    for match in _VALUE.finditer(text):
        value = _value(match)
        starts[match.start()] = ends[match.end()] = value
        if match["year"]:
            # A more specific date also preserves a month/day or a year supplied on its own.
            anchors.add(value[:-1])
            anchors.add(("token", match["year"]))
        if value[0] != "token" or (
            any(c.isdigit() for c in value[1]) and not _ORDINAL.fullmatch(value[1])
        ):
            anchors.add(value)
        if value[0] == "token" and (equals := _EQUALS.match(text, match.end())):
            start = equals.end()
            if quoted := _QUOTED_VALUE.match(text, start):
                anchors.add(("assignment", value[1], "string", quoted.group()[1:-1]))
            elif assigned := _VALUE.match(text, start):
                anchors.add(("assignment", value[1], *_value(assigned)))
    for quoted in _QUOTED_OPERAND.finditer(text):
        raw = quoted.group()[1:-1]
        inner = _VALUE.fullmatch(raw)
        inner_value = _value(inner) if inner else None
        operand = (
            inner_value
            if inner_value and _is_bound(inner_value)
            else ("quoted-field", raw.casefold())
        )
        starts[quoted.start()] = ends[quoted.end()] = operand
    anchors.update(_comparisons(text, starts, ends))
    return anchors


def _named_codes(text: str, keys: dict[int, str], *, literal: bool) -> set[Anchor]:
    codes: set[Anchor] = set()
    words = list(_WORD.finditer(text))
    for index, (left, right) in enumerate(pairwise(words)):
        if not text[left.end() : right.start()].isspace():
            continue
        if left.group().casefold() in _KINDS:
            kind, code = left, right
        elif right.group().casefold() in _KINDS:
            kind, code = right, left
        else:
            continue
        if kind.start() in keys:
            continue
        raw = code.group()
        spelling = raw.strip(_QUOTES)
        # Digit-bearing identifiers are already protected even if the rewrite omits the noun.
        if any(c.isdigit() for c in spelling):
            continue
        if raw == spelling:
            if literal and (not spelling.isupper() or spelling == "I"):
                continue
            if code is left and spelling.casefold() == "a":
                if not index or words[index - 1].group().casefold() != "the":
                    continue
            elif spelling == "a" and not _AFTER_A.match(text, code.end()):
                continue
        codes.add((kind.group().casefold(), spelling.casefold()))
    return codes


def _following_key(text: str, key: re.Match[str]) -> re.Match[str] | None:
    join = _LIST_JOIN.match(text, key.end())
    return _KEY_ITEM.match(text, join.end()) if join else None


def _declared_keys(text: str, *, require_marker: bool) -> dict[int, str]:
    """Index each declared name by its position so a key such as Region is not a type label."""
    names: dict[int, str] = {}
    case_sensitive = not require_marker or bool(_CASE_SENSITIVE.search(text))
    labels = [
        m
        for m in _KEY_LABEL.finditer(text)
        if case_sensitive or "json" in m.group().casefold().split()
    ]
    if not labels:
        return names
    # Bind qualified declarations first. In "name JSON keys", name belongs to the list
    # and must not first consume JSON as the value of an unrelated "name" declaration.
    labels.sort(key=lambda label: len(label.group().split()) == 1)
    # Lists may precede or follow their label. Keep quoted names atomic, even with a comma inside.
    before_label: dict[int, re.Match[str]] = {}
    preceding: dict[int, re.Match[str]] = {}
    coordinated: set[int] = set()
    for item in _KEY_ITEM.finditer(text):
        if space := _SPACE.match(text, item.end()):
            before_label[space.end()] = item
        if next_item := _following_key(text, item):
            preceding[next_item.start()] = item
            if text[item.end() : next_item.start()].strip() != ",":
                coordinated.add(next_item.start())
    for label in labels:
        if label.start() in names:
            continue
        before = before_label.get(label.start())
        intro = _NAME_INTRO.match(text, label.end())
        after = _KEY_ITEM.match(text, intro.end()) if intro else None
        preposed = before is not None and (
            _looks_like_key(before.group()) or before.start() in preceding
        )
        if after and intro:
            # A bare function word in "Which JSON key is used?" supplies no identifier.
            after_in_list = _following_key(text, after)
            if not intro[1] and after.group() in ENGLISH_STOPWORDS and not after_in_list:
                after = None
            elif intro[1] or _looks_like_key(after.group()) or after_in_list:
                preposed = False
        key = before if preposed else after
        if key is None:
            continue
        plural = label.group().casefold().endswith("s")
        raw = key.group()
        names[key.start()] = raw[1:-1] if raw[0] in _QUOTES else raw
        completed_list = False
        while key := (preceding.get(key.start()) if preposed else _following_key(text, key)):
            raw = key.group()
            if raw[0] not in _QUOTES:
                next_label = _KEY_LABEL.match(text, key.start())
                # A multiword label starts another declaration. A lone "name" can itself be a key.
                starts_label = next_label is not None and next_label.end() > key.end()
                # After "A and B", a second conjunction may introduce a new request. Keep
                # a further bare name only when a list delimiter or boundary follows it.
                starts_clause = (
                    completed_list
                    and key.start() in coordinated
                    and not _LIST_END.match(text, key.end())
                    and not _following_key(text, key)
                )
                if starts_label or (
                    raw.islower() and raw.isalpha() and (not plural or starts_clause)
                ):
                    break
            names[key.start()] = raw[1:-1] if raw[0] in _QUOTES else raw
            if not preposed and key.start() in coordinated:
                completed_list = True
    return names


def _looks_like_key(raw: str) -> bool:
    """Recognize a quoted name or an identifier with capitals, digits or punctuation."""
    return raw[0] in _QUOTES or (
        raw.casefold() not in ENGLISH_STOPWORDS
        and (raw.isupper() or not raw.isalpha() or any(c.isupper() for c in raw[1:]))
    )


def preserves_explicit_anchors(literal: str, proposed: str) -> bool:
    """Compare structured values first, and preserve exact spelling where the caller supplied it."""
    if not _constraints(literal) <= _constraints(proposed):
        return False
    literal_keys = _declared_keys(literal, require_marker=True)
    proposed_keys = _declared_keys(proposed, require_marker=not literal_keys)
    if not _named_codes(literal, literal_keys, literal=True) <= _named_codes(
        proposed, proposed_keys, literal=False
    ):
        return False
    exact = set(literal_keys.values())
    if not exact:
        return True
    if proposed_keys:
        return exact <= set(proposed_keys.values())
    # A standalone question may use the exact name without repeating its type.
    spellings = {m.group() for m in re.finditer(_IDENTIFIER, proposed)}
    spellings.update(m.group()[1:-1] for m in _QUOTED_VALUE.finditer(proposed))
    return exact <= spellings
