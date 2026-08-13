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

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/), a running A2A agent. For trace analysis, a running MLflow server is required.

```bash
uv sync
```

Run the full pipeline:

```bash
agenttrust pipeline http://localhost:8000 --num-probes 5
```

To use agent names instead of URLs, add your agent to `agents.yaml`:

```yaml
weather_agent: http://localhost:8000
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

| Flag | Env var | Default | Description |
|------|--------|---------|-------------|
| `<agent>` | `AGENT_URL` | — | Agent URL or name |
| `--num-probes` | `AGENTTRUST_NUM_PROBES` | `5` | Probes per scope |
| `--gen-model` | `AGENTTRUST_GEN_MODEL` | `claude-haiku-4-5` | LLM for prompt generation |
| `--judge-model` | `AGENTTRUST_JUDGE_MODEL` | `claude-haiku-4-5` | LLM for evaluation judge |
| `--trace-source` | `AGENTTRUST_TRACE_SOURCE` | `none` | `mlflow` or `none` |
| `--experiment` | `MLFLOW_EXPERIMENT_NAME` | `agent-trust` | MLflow experiment name |
| `--work-dir` | `AGENTTRUST_WORK_DIR` | `.` | Working directory for pipeline output |
| `--probe-timeout` | `AGENTTRUST_PROBE_TIMEOUT` | `120` | Per-probe timeout in seconds |
| `--job-deadline` | `AGENTTRUST_JOB_DEADLINE` | `900` | Overall pipeline deadline in seconds |
| `--alignment-threshold` | `AGENTTRUST_ALIGNMENT_THRESHOLD` | `70.0` | Score threshold for alignment pass |
| `--verbose` | | | Enable debug logging |
| — | `MLFLOW_TRACKING_URI` | — | MLflow server URI for report upload |

CLI flags take priority over environment variables. Agent names are resolved via `agents.yaml`; URLs are accepted directly.

## Docker

```bash
docker build -t agenttrust .
docker run --rm -e AGENT_URL=http://host.docker.internal:8000 -e ANTHROPIC_API_KEY agenttrust
```
