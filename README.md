# AgentTrust

Evaluate what AI agents do vs. what they say.

AI agents declare what they can do, but behavior can diverge from intent, and declarations alone mean nothing without verification.
AgentTrust evaluates an agent and produces a trust score based on how closely its behavior aligns with its declared capabilities. It reads the agent's card and generates prompts across three scopes:

- **in-scope** — requests the agent should handle according to its declared skills
- **out-of-scope** — requests clearly unrelated to the agent's declared capabilities
- **near-miss** — requests in the agent's domain but outside its declared skills

Each response is scored by an LLM judge on whether the agent stayed within its declared skills. When the agent is
instrumented with MLflow, runtime traces are cross-referenced to detect tool-use
violations. The result is a trust score report.

## Quick Start

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/), a running A2A agent.

```bash
uv sync
```

Run the full pipeline:

```bash
agenttrust pipeline http://localhost:8002 --num-probes 5
```

To use agent names instead of URLs, add your agent to `agents.yaml`:

```yaml
weather_agent: http://localhost:8002
```

```bash
agenttrust pipeline weather_agent --num-probes 5
```

If the agent is instrumented with MLflow, enable trace analysis to detect tool-use violations:

```bash
agenttrust pipeline weather_agent --num-probes 5 --trace-source mlflow
```

## How It Works

1. **Generate** — LLM generates realistic prompts per scope from the agent's card
2. **Probe** — sends prompts to the agent via A2A protocol, records responses with timing
3. **Evaluate** — LLM judge scores each response for scope compliance (0.0-1.0)
4. **Report** — merges text scores with trace verdicts, computes weighted trust score

## CLI Reference

```
agenttrust pipeline <agent> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--num-probes` | 5 | Probes per scope |
| `--gen-model` | `claude-haiku-4-5` | LLM for prompt generation |
| `--judge-model` | `claude-haiku-4-5` | LLM for evaluation judge |
| `--trace-source` | `none` | `mlflow` or `none` |
| `--experiment` | `agent-trust` | MLflow experiment name |
| `--verbose` | | Enable debug logging |

Agent names are resolved via `agents.yaml`; URLs are accepted directly.
