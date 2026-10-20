"""Tests for the optional menu/order workflow."""

import json
import os
import tempfile
import unittest

from orders import Menu, MenuItem, OrderBook


class TestMenu(unittest.TestCase):
    def menu(self):
        return Menu(
            [
                MenuItem("Cheeseburger", aliases=["cheese burger", "burger"], price=9.5),
                MenuItem("Coke", aliases=["cola"], price=2.0),
                MenuItem("Fries", price=3.0),
            ]
        )

    def test_resolve_exact_and_alias(self):
        menu = self.menu()
        self.assertEqual(menu.resolve("cheeseburger").name, "Cheeseburger")
        self.assertEqual(menu.resolve("Cheese Burger").name, "Cheeseburger")
        self.assertIsNone(menu.resolve("pizza"))

    def test_find_all_uses_longest_non_overlapping_alias(self):
        matches = self.menu().find_all("one cheeseburger and a coke")
        self.assertEqual([match[2].name for match in matches], ["Cheeseburger", "Coke"])

    def test_plural_menu_items_match(self):
        matches = self.menu().find_all("two cheeseburgers")
        self.assertEqual(matches[0][2].name, "Cheeseburger")

    def test_from_file_reads_yaml_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = os.path.join(tmp, "menu.yaml")
            json_path = os.path.join(tmp, "menu.json")
            with open(yaml_path, "w", encoding="utf-8") as handle:
                handle.write("items:\n  - name: Burger\n    price: 5\n")
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump({"items": [{"name": "Burger", "price": 5}]}, handle)
            self.assertEqual(Menu.from_file(yaml_path).resolve("burger").price, 5)
            self.assertEqual(Menu.from_file(json_path).resolve("burger").price, 5)


class TestOrderBook(unittest.TestCase):
    def setUp(self):
        self.book = OrderBook(
            Menu(
                [
                    MenuItem("Cheeseburger", aliases=["burger"], price=9.5),
                    MenuItem("Coke", aliases=["cola"], price=2.0),
                ]
            )
        )

    def test_add_text_parses_quantities_and_plurals(self):
        added = self.book.add_text("4", "two cheeseburgers and a coke")
        self.assertEqual([(item.name, item.quantity) for item in added], [("Cheeseburger", 2), ("Coke", 1)])
        self.assertEqual(self.book.summary("4"), "Table 4: 2 x Cheeseburger, 1 x Coke — total 21.00")

    def test_modifiers_attach_to_the_preceding_item(self):
        self.book.add_text("4", "two cheeseburgers no pickles and a coke")
        order = self.book.get("4")
        self.assertEqual(order.items[0].modifiers, ["no pickles"])
        self.assertEqual(order.items[1].modifiers, [])

    def test_add_item_merges_the_same_item_and_modifiers(self):
        self.book.add_item("4", "Cheeseburger", quantity=1, modifiers=["no pickles"])
        self.book.add_item("4", "burger", quantity=2, modifiers=["no pickles"])
        order = self.book.get("4")
        self.assertEqual(len(order.items), 1)
        self.assertEqual(order.items[0].quantity, 3)
        self.assertEqual(order.items[0].price, 9.5)

    def test_confirm_and_close(self):
        self.book.add_item("4", "Coke")
        confirmed = self.book.confirm("4")
        self.assertTrue(confirmed.confirmed)
        self.assertEqual(confirmed.phase, "confirmed")
        closed = self.book.close("4")
        self.assertIs(closed, confirmed)
        self.assertIsNone(self.book.get("4"))

    def test_to_dict_is_json_friendly(self):
        self.book.add_item("4", "Coke", 2)
        data = self.book.get("4").to_dict()
        self.assertEqual(data["table"], "4")
        self.assertEqual(data["items"][0]["quantity"], 2)
        self.assertEqual(data["total"], 4.0)


if __name__ == "__main__":
    unittest.main()
