"""Optional structured order intake for restaurant/retail workflows.

This module is intentionally separate from the attention gate.  The gate can
keep other-directed speech in its `other` buffer; an application can feed that
text to an :class:`OrderBook`, or use the LLM to produce structured items and
call ``add_item`` directly.

The text parser is deliberately small and deterministic: it matches menu names
and aliases, quantities and simple modifiers.  Use the LLM for messy language;
use this module to validate and keep the resulting order structured.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().rstrip(".!?,").split())


_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "a": 1,
    "an": 1,
}
_QUANTITY_RE = re.compile(r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|a|an)\b", re.IGNORECASE)
_MODIFIER_RE = re.compile(
    r"\b(no|without|extra)\s+([a-zA-Z]+(?:\s+(?!and\b|with\b)[a-zA-Z]+)?)",
    re.IGNORECASE,
)


@dataclass
class MenuItem:
    name: str
    aliases: List[str] = field(default_factory=list)
    price: Optional[float] = None
    modifiers: List[str] = field(default_factory=list)


@dataclass
class OrderItem:
    name: str
    quantity: int = 1
    modifiers: List[str] = field(default_factory=list)
    price: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"name": self.name, "quantity": self.quantity, "modifiers": list(self.modifiers)}
        if self.price is not None:
            data["price"] = self.price
        return data


@dataclass
class Order:
    table: str
    items: List[OrderItem] = field(default_factory=list)
    phase: str = "order"
    confirmed: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum((item.price or 0.0) * item.quantity for item in self.items)

    def summary(self) -> str:
        if not self.items:
            return f"Table {self.table}: no items"
        parts = []
        for item in self.items:
            detail = f"{item.quantity} x {item.name}"
            if item.modifiers:
                detail += f" ({', '.join(item.modifiers)})"
            parts.append(detail)
        summary = f"Table {self.table}: " + ", ".join(parts)
        if any(item.price is not None for item in self.items):
            summary += f" — total {self.total:.2f}"
        return summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "items": [item.to_dict() for item in self.items],
            "phase": self.phase,
            "confirmed": self.confirmed,
            "notes": list(self.notes),
            "total": self.total,
        }


class Menu:
    def __init__(self, items: Iterable[MenuItem]):
        self.items = list(items)

    @classmethod
    def from_dict(cls, data: Any) -> "Menu":
        entries = data.get("items", []) if isinstance(data, dict) else data
        if not isinstance(entries, list):
            raise ValueError("menu must be a list of items or an object with an 'items' list")
        items = []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("name"):
                raise ValueError("each menu item needs a name")
            items.append(
                MenuItem(
                    name=str(entry["name"]),
                    aliases=[str(alias) for alias in entry.get("aliases", [])],
                    price=float(entry["price"]) if entry.get("price") is not None else None,
                    modifiers=[str(modifier) for modifier in entry.get("modifiers", [])],
                )
            )
        return cls(items)

    @classmethod
    def from_file(cls, path: str) -> "Menu":
        with open(path, "r", encoding="utf-8") as handle:
            if path.lower().endswith(".json"):
                data = json.load(handle)
            else:
                data = yaml.safe_load(handle)
        return cls.from_dict(data or [])

    def resolve(self, name: str) -> Optional[MenuItem]:
        target = _normalize(name)
        for item in self.items:
            names = [item.name, *item.aliases]
            if any(_normalize(candidate) == target for candidate in names):
                return item
        return None

    def find_all(self, text: str) -> List[Tuple[int, int, MenuItem, str]]:
        """Longest, non-overlapping menu matches: (start, end, item, matched alias)."""
        haystack = text.lower()
        candidates: List[Tuple[int, int, MenuItem, str]] = []
        for item in self.items:
            for alias in [item.name, *item.aliases]:
                needle = alias.lower().strip()
                if not needle:
                    continue
                for match in re.finditer(rf"(?<!\w){re.escape(needle)}s?(?!\w)", haystack):
                    candidates.append((match.start(), match.end(), item, needle))
        candidates.sort(key=lambda entry: (entry[0], -(entry[1] - entry[0])))
        chosen: List[Tuple[int, int, MenuItem, str]] = []
        for candidate in candidates:
            start, end = candidate[0], candidate[1]
            if all(end <= existing[0] or start >= existing[1] for existing in chosen):
                chosen.append(candidate)
        return chosen


class OrderBook:
    def __init__(self, menu: Optional[Menu] = None):
        self.menu = menu
        self._orders: Dict[str, Order] = {}

    @property
    def orders(self) -> List[Order]:
        return list(self._orders.values())

    def get(self, table: str) -> Optional[Order]:
        return self._orders.get(table)

    def start(self, table: str) -> Order:
        order = self._orders.get(table)
        if order is None:
            order = Order(table=table)
            self._orders[table] = order
        return order

    def add_item(
        self,
        table: str,
        name: str,
        quantity: int = 1,
        modifiers: Optional[List[str]] = None,
    ) -> OrderItem:
        if quantity < 1:
            raise ValueError("quantity must be at least 1")
        order = self.start(table)
        menu_item = self.menu.resolve(name) if self.menu else None
        canonical = menu_item.name if menu_item else name
        item = OrderItem(
            name=canonical,
            quantity=quantity,
            modifiers=list(modifiers or []),
            price=menu_item.price if menu_item else None,
        )
        for existing in order.items:
            if existing.name == item.name and existing.modifiers == item.modifiers:
                existing.quantity += item.quantity
                return existing
        order.items.append(item)
        return item

    def add_text(self, table: str, text: str) -> List[OrderItem]:
        """Parse simple menu text; returns the items it added.

        This is a convenience for clean, menu-matched speech.  For messy natural
        language, extract items with the LLM and call :meth:`add_item`.
        """
        if self.menu is None:
            return []
        matches = self.menu.find_all(text)
        if not matches:
            return []
        order = self.start(table)
        added: List[OrderItem] = []
        for start, _end, menu_item, _alias in matches:
            prefix = text[:start]
            quantities = list(_QUANTITY_RE.finditer(prefix))
            quantity = 1
            if quantities:
                raw = quantities[-1].group(1).lower()
                quantity = int(raw) if raw.isdigit() else _NUMBERS.get(raw, 1)
            added_item = self.add_item(table, menu_item.name, quantity)
            added.append(added_item)

        for match in _MODIFIER_RE.finditer(text):
            phrase = f"{match.group(1).lower()} {match.group(2).strip().lower()}"
            target = None
            for start, _end, menu_item, _alias in matches:
                if start <= match.start():
                    target = menu_item
            if target is None:
                if phrase not in order.notes:
                    order.notes.append(phrase)
            else:
                target_item: Optional[OrderItem] = None
                for candidate in order.items:
                    if candidate.name == target.name:
                        target_item = candidate
                        break
                if target_item is not None and phrase not in target_item.modifiers:
                    target_item.modifiers.append(phrase)
        return added

    def confirm(self, table: str) -> Order:
        order = self.start(table)
        order.confirmed = True
        order.phase = "confirmed"
        return order

    def close(self, table: str) -> Optional[Order]:
        return self._orders.pop(table, None)

    def clear(self) -> None:
        self._orders.clear()

    def summary(self, table: str) -> Optional[str]:
        order = self._orders.get(table)
        return order.summary() if order else None
