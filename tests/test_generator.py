import json

from agenttrust.generator import build_scope_instruction, save_prompts


class TestBuildScopeInstruction:
    def test_previous_prompts_appended(self) -> None:
        result = build_scope_instruction("in_scope", ["what's the weather?", "is it raining?"])
        assert "what's the weather?" in result
        assert "is it raining?" in result
        assert "Do NOT repeat" in result


class TestSavePrompts:
    def test_writes_jsonl(self, tmp_path) -> None:
        output = tmp_path / "sub" / "prompts.jsonl"
        save_prompts(["hello", "world"], output)

        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"prompt": "hello"}
        assert json.loads(lines[1]) == {"prompt": "world"}
