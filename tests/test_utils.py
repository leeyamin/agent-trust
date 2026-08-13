from agenttrust.utils import compute_card_hash, strip_markdown_fences


class TestStripMarkdownFences:
    def test_plain_text_unchanged(self) -> None:
        assert strip_markdown_fences("hello world") == "hello world"

    def test_strips_json_fenced_block(self) -> None:
        text = '```json\n{"key": "value"}\n```'
        assert strip_markdown_fences(text) == '{"key": "value"}'


class TestComputeCardHash:
    def test_different_key_order_produces_same_hash(self) -> None:
        card_a = {"name": "weather", "skills": []}
        card_b = {"skills": [], "name": "weather"}
        assert compute_card_hash(card_a) == compute_card_hash(card_b)
