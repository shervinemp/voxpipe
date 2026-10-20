# Restaurant example

This is an **application example**, not part of the `voxpipe` core. It shows how
to put the generic attention gate to work in a restaurant while keeping menu,
table and order logic outside the voice-assistant package.

The core package knows about:

- `ConversationGate` / `AddresseeGate`
- topic and side buffers
- shadow decision logging and replay
- deferred replies

It does **not** know about:

- menus,
- orders,
- prices,
- tables,
- POS confirmation.

Those live here.

## Run the order tests

```bash
uv run pytest examples/restaurant
```

## Use the order book

```python
from orders import Menu, OrderBook

menu = Menu.from_file("menu.example.yaml")
orders = OrderBook(menu)

orders.add_text("4", "two cheeseburgers no pickles and a coke")
print(orders.summary("4"))
# Table 4: 2 x Cheeseburger (no pickles), 1 x Coke — total 21.00

orders.confirm("4")
orders.close("4")
```

For messy natural language, let the LLM extract structured items and call
`orders.add_item(table, name, quantity, modifiers)` rather than relying on the
small deterministic parser. Expose those operations as `voxpipe` tools:

```python
@Tool.from_callable
def add_order_item(table, item, quantity=1, modifiers=None):
    """Add an item to a table order."""
    return orders.add_item(table, item, quantity, modifiers).to_dict()
```

That keeps the voice pipeline generic and makes the restaurant workflow an
application built on top of it.

## Busy mode

Enable the generic addressee gate:

```yaml
attention:
  busy_mode: true
  shadow: true
  decision_log_path: "data/attention_decisions.jsonl"
```

Use `attention.busy_mode` as the voice-assistant preset; use this example only
for the restaurant-specific order layer.
