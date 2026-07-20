import pytest

from agenttrust.cli import _get_agent_url


class TestResolveAgentUrl:
    def test_name_resolved_from_yaml(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = tmp_path / "agents.yaml"
        config.write_text("weather_agent: http://localhost:8002\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        assert _get_agent_url("weather_agent") == "http://localhost:8002"

    def test_unknown_name_raises(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = tmp_path / "agents.yaml"
        config.write_text("weather_agent: http://localhost:8002\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        with pytest.raises(SystemExit, match="not found in agents.yaml"):
            _get_agent_url("wiki_agent")
