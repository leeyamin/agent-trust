from agenttrust.utils import strip_markdown_fences


class TestStripMarkdownFences:
    def test_plain_text_unchanged(self) -> None:
        assert strip_markdown_fences("hello world") == "hello world"

    def test_strips_json_fenced_block(self) -> None:
        text = '```json\n{"key": "value"}\n```'
        assert strip_markdown_fences(text) == '{"key": "value"}'
