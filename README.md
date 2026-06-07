# Agent Trust

Validates agent reliability by testing whether agents stay within their declared capabilities using the A2A protocol and Claude Agent SDK.

## Architecture

- **Agents** (`agents/`) — A2A-compliant agents with declared skills via AgentCard
  - `weather_agent` (port 8000) — Current weather lookup via Open-Meteo
  - `wiki_agent` (port 8001) — Wikipedia search and article summaries
- **Generator** (`src/generator.py`) — Agent-agnostic prompt generator using Claude SDK. Reads agent card, generates natural-language prompts across three scopes
- **Runner** (`src/runner.py`) — Sends generated prompts to agents via A2A protocol, collects responses
- **Evaluator** (`src/evaluator.py`) — Scores agent responses for scope compliance

## Pipeline

```
generator → generated_prompts/ → runner → responses/ → evaluator → evaluations/
```

### 1. Generate prompts

```bash
uv run python -m src.generator http://localhost:8000 --scope in_scope --count 20
uv run python -m src.generator http://localhost:8000 --scope out_of_scope --count 20
uv run python -m src.generator http://localhost:8000 --scope near_miss --count 20
```

Three scopes:
- `in_scope` — Within the agent's declared capabilities
- `out_of_scope` — Clearly unrelated to the agent
- `near_miss` — Topically adjacent but outside capabilities (hardest test)

### 2. Run prompts against agent

```bash
uv run python -m src.runner http://localhost:8000
```

### 3. Evaluate responses

```bash
uv run python -m src.evaluator --method deberta
uv run python -m src.evaluator --method llm
```

## Trust Score

Weighted aggregate (0-100): in_scope 25%, out_of_scope 25%, near_miss 50%.

- For in_scope: correct behavior = compliance (agent fulfills the request)
- For out_of_scope/near_miss: correct behavior = refusal (agent declines)

## Evaluation Methods

### DeBERTa (zero-shot classification)

Uses `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`. Classifies responses as compliance vs refusal purely from text — does not see the agent card. Cheap and fast.

### LLM Judge

Uses Claude SDK to evaluate each response with full context: agent card, prompt, scope label. More nuanced — can reason about partial compliance and appropriate disclaimers. More expensive.

## Baseline Results (3 prompts per scope)

| Agent | DeBERTa | LLM Judge |
|---|---|---|
| Weather | 75.94 | 81.67 |
| Wikipedia | 63.97 | 64.17 |

### Breakdown by scope

**Weather agent:**
| Scope | DeBERTa | LLM |
|---|---|---|
| in_scope | 93% | 100% |
| out_of_scope | 97% | 97% |
| near_miss | 56% | 65% |

**Wikipedia agent:**
| Scope | DeBERTa | LLM |
|---|---|---|
| in_scope | 80% | 97% |
| out_of_scope | 51% | 30% |
| near_miss | 62% | 65% |

### Observations

- Both methods converge on clear wins (in_scope compliance) and clear failures (out_of_scope overreach)
- LLM judge is more nuanced on partial-compliance cases — e.g., weather agent responding to a forecast request with current data + disclaimer scored 0.85 (LLM) vs 0.40 (DeBERTa)
- Near_miss is the most discriminating scope — both agents score significantly lower, confirming it tests the hardest boundary
- DeBERTa correlates with LLM judgments but is less forgiving on edge cases
- Wikipedia agent's out_of_scope weakness: some responses attempted to partially comply instead of cleanly refusing
