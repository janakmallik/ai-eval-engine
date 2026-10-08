# AI Eval Engine

**AI evaluation, regression testing, and observability infrastructure for AI/ML systems.**

`ai-eval-engine` is a Python framework for evaluating AI applications and understanding whether a new model, prompt, RAG pipeline, or agent version actually improved.

It combines **evaluation, experiment configuration, regression detection, quality gates, and execution tracing** into a single engineering workflow.

The core idea is:

```text
AI Application
      │
      ▼
Evaluation
      │
      ├── Model
      ├── Retrieval
      ├── Tools
      └── Evaluators
      │
      ▼
Evaluation Run
      │
      ├── Metrics
      ├── Experiment Identity
      ├── Regression Detection
      └── Trace
      │
      ▼
Reports / CI Quality Gates
```

---

## Why This Exists

AI systems are difficult to improve safely.

Changing a model, prompt, retrieval configuration, or agent workflow can improve one dimension while silently degrading another.

For example:

```text
                    BEFORE     AFTER

Answer quality       0.91       0.94    ✓
Retrieval recall     0.88       0.91    ✓
Latency              0.80s      1.40s   ⚠
```

A production-oriented evaluation system should answer:

> **Did the new version actually get better, and if not, what changed?**

`ai-eval-engine` is designed around that problem.

---

# Features

## Evaluation

Evaluate a model or AI pipeline against a reproducible dataset.

Core abstractions include:

- `EvalCase`
- `EvalDataset`
- `EvaluationContext`
- `EvaluationResult`
- `EvaluationRun`

Built-in evaluators include:

- Exact match
- Contains
- Similarity
- Length
- Classification
- LLM judge
- Retrieval contains
- Retrieval precision
- Retrieval recall

Evaluators operate against a common evaluation interface, making the engine extensible with additional evaluation strategies.

---

## Configuration-Based Evaluation

Evaluation configurations provide a structured definition of an experiment.

```python
from aieval import EvaluationConfig, evaluate_with_config

config = EvaluationConfig(
    model="qa-model",
    model_version="v2",
    dataset="capitals-v1",
)

run = evaluate_with_config(
    config,
    model=my_model,
    dataset=my_dataset,
)
```

A configuration can describe:

- model
- model version
- model provider
- dataset
- dataset version
- prompt
- prompt version
- retrieval `top_k`
- evaluator configuration
- metadata
- tracing configuration

Configurations receive a deterministic `config_id`, allowing experiment definitions to be identified reproducibly.

---

# Experiments

Experiments connect a configuration with an evaluation run.

```text
Experiment
├── Configuration
│   ├── Model
│   ├── Dataset
│   ├── Prompt
│   └── Evaluators
│
└── Evaluation Run
    ├── Results
    ├── Metrics
    └── Trace
```

Experiments can be persisted and retrieved through `ExperimentStore`.

Supported operations include:

- save
- load
- list
- find
- delete
- overwrite
- metadata preservation
- evaluator configuration preservation
- trace preservation

Experiment identity is derived from the experiment definition/configuration rather than being an arbitrary identifier.

The experiment store also supports schema versioning and legacy compatibility.

---

# Regression Testing

The engine can compare evaluation runs to detect regressions.

Core components include:

- `ComparisonResult`
- `RegressionConfig`
- `RegressionDetector`
- `RegressionResult`
- `RegressionGate`
- `GateResult`

Regression analysis can evaluate changes between a baseline and current run.

Example:

```text
                 BASELINE     CURRENT

Quality             0.91        0.94
Latency             0.80s       1.40s
```

Thresholds can be configured to determine whether a change should be considered acceptable.

Quality gates expose a PASS/FAIL result that can be used by automated workflows.

The CLI also supports regression comparisons against JSON evaluation reports.

---

# RAG Evaluation

`ai-eval-engine` does **not** attempt to be a complete RAG application or vector database.

Instead, it provides evaluation and observability infrastructure that can be integrated into RAG systems.

Retrieval-specific evaluators include:

- `RetrievalContainsEvaluator`
- `RetrievalPrecisionEvaluator`
- `RetrievalRecallEvaluator`

These allow retrieval quality to be evaluated independently from the final generated answer.

A RAG workflow can therefore be analyzed at multiple levels:

```text
Query
  │
  ▼
Retrieval
  │
  ├── Retrieval quality
  ├── Retrieved result count
  └── Retrieval latency
  │
  ▼
Model
  │
  ▼
Generated answer
  │
  ▼
Answer evaluation
```

---

# Tracing & Observability

The engine provides execution tracing for AI workflows.

Core tracing components include:

- `Trace`
- `Span`
- `ModelResponse`
- `ModelUsage`
- `ToolResponse`

Traces support hierarchical execution structures.

For example:

```text
evaluation
└── evaluation.case
    ├── retrieval
    ├── model
    ├── evaluator.exact_match
    └── tool
```

Depending on the application, a trace can contain different combinations of these operations.

Spans record information such as:

- start time
- end time
- duration
- attributes
- parent/child relationships
- lifecycle
- status
- exceptions

This makes it possible to inspect not only whether an evaluation passed, but also what happened while the AI system was executing.

---

## Retrieval Tracing

Retrieval operations can be represented as dedicated trace spans.

A retrieval span can record information including:

- query
- `top_k`
- result count
- latency

Retrieval spans participate in the same trace hierarchy as model and evaluator spans.

---

## Model Tracing

Model execution can be represented as a model span containing model request/response information and usage data.

The tracing layer supports model usage information such as token usage where provided by the model integration.

---

## Tool Tracing

Tool execution can also be represented within the trace hierarchy.

This is particularly useful for agent-style workflows where execution may involve:

```text
Agent
 ├── Model
 ├── Tool
 │    └── External operation
 ├── Retrieval
 └── Model
```

---

# JSON Reporting

Evaluation runs can be serialized to JSON using `JsonReporter`.

```python
from aieval.reporting.json import JsonReporter

reporter = JsonReporter()

reporter.write(run, "evaluation.json")

loaded_run = reporter.read("evaluation.json")
```

JSON reports preserve evaluation information including:

- results
- metadata
- run identity
- aggregate metrics
- evaluator summaries
- trace information
- retrieval traces
- nested trace hierarchy

This allows evaluation results to be stored, compared, and consumed by CLI or CI workflows.

---

# Command-Line Interface

The package provides an `aieval` command.

```bash
aieval --help
```

Current commands:

```text
aieval regression
aieval trace
```

### Regression

Compare evaluation reports and determine whether configured thresholds have been violated.

```bash
aieval regression ...
```

### Trace

Inspect the execution trace stored in an evaluation JSON report.

```bash
aieval trace ...
```

The trace command can render the trace hierarchy and span information, including attributes and error information.

---

# Public Python API

The main package exposes the core functionality through `aieval`.

Examples:

```python
from aieval import (
    EvalCase,
    EvalDataset,
    EvaluationConfig,
    EvaluationContext,
    EvaluationResult,
    EvaluationRun,
    ExactMatchEvaluator,
    ContainsEvaluator,
    SimilarityEvaluator,
    LengthEvaluator,
    Experiment,
    ExperimentStore,
    RegressionConfig,
    RegressionDetector,
    RegressionGate,
    Trace,
    evaluate_dataset,
    evaluate_with_config,
)
```

The public API is intended to keep application code independent from the internal package structure where practical.

---

# Installation

The project requires Python 3.12 or newer.

From the repository:

```bash
git clone https://github.com/janakmallik/ai-eval-engine.git
cd ai-eval-engine

python -m venv .venv
```

Activate the environment and install:

```bash
pip install -e .
```

The package also defines the `aieval` command-line executable.

---

# Development

Install the project in editable mode:

```bash
python -m pip install -e .
```

Run the test suite:

```bash
pytest -q
```

The project is developed using an incremental TDD workflow:

```text
Requirement
    ↓
Failing test
    ↓
Implementation
    ↓
Targeted tests
    ↓
Full test suite
    ↓
Commit
    ↓
Push
```

---

# Architecture

At a high level:

```text
                    ┌────────────────────┐
                    │    AI Application  │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Evaluation Runner  │
                    └─────────┬──────────┘
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
        Retrieval           Model           Tools
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                    ┌────────────────────┐
                    │     Evaluators     │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │  Evaluation Run    │
                    └─────────┬──────────┘
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
        Experiments      Regression          Trace
             │             Detection            │
             │                │                  │
             └────────────────┼──────────────────┘
                              ▼
                    ┌────────────────────┐
                    │ JSON / CLI / CI    │
                    └────────────────────┘
```

---

# Project Structure

```text
ai-eval-engine/
│
├── src/
│   └── aieval/
│       ├── cli.py
│       ├── comparison.py
│       ├── config.py
│       ├── context.py
│       ├── dataset.py
│       ├── experiment.py
│       ├── gate.py
│       ├── normalizers.py
│       ├── regression.py
│       ├── result.py
│       ├── run.py
│       ├── runner.py
│       ├── store.py
│       ├── summary.py
│       │
│       ├── evaluators/
│       │   ├── base.py
│       │   ├── classification.py
│       │   ├── contains.py
│       │   ├── exact_match.py
│       │   ├── length.py
│       │   ├── llm_judge.py
│       │   ├── retrieval_contains.py
│       │   ├── retrieval_precision.py
│       │   ├── retrieval_recall.py
│       │   └── similarity.py
│       │
│       ├── reporting/
│       │   └── json.py
│       │
│       └── tracing/
│           ├── response.py
│           ├── span.py
│           ├── tool.py
│           ├── trace.py
│           └── usage.py
│
├── examples/
│   ├── basic.py
│   ├── experiment_demo.py
│   ├── regression_demo.py
│   └── trace_demo.py
│
├── tests/
│
├── pyproject.toml
└── README.md
```

---

# Current Status

The current implementation contains the major foundation for:

- deterministic evaluation
- evaluator-based scoring
- evaluation runs
- configuration-driven evaluation
- experiment management
- experiment persistence
- run comparison
- regression detection
- quality gates
- JSON reporting
- model tracing
- retrieval tracing
- tool tracing
- exception observability
- RAG-specific evaluation
- CLI-based regression and trace inspection

The project currently has a comprehensive automated test suite covering the implemented functionality.

---

# Roadmap

The original roadmap is structured around progressively expanding the evaluation and observability system.

### V1 — Evaluation Engine

- Evaluation cases and datasets
- Evaluator abstraction
- Built-in evaluators
- Evaluation runs
- Result and metric aggregation

**Status: Complete**

### V2 — Experiments & Configuration

- Evaluation configuration
- Experiment identity
- Experiment persistence
- Run comparison
- Metadata
- Schema/version handling

**Status: Complete**

### V3 — Tracing & Observability

- Hierarchical traces
- Model spans
- Retrieval spans
- Tool spans
- Evaluator spans
- Error/exception observability
- Trace serialization
- Trace inspection

**Status: Complete / substantially complete**

### V4 — LLM Cost & Performance

Planned dedicated analytics layer:

- input tokens
- output tokens
- total tokens
- model name
- latency
- estimated cost
- errors
- retries
- timeouts
- p50 latency
- p95 latency
- average latency
- average tokens
- cost/request
- error rate

**Status: Deferred**

### V5 — Regression Testing

Planned/implemented capabilities include:

- baseline comparison
- current-run comparison
- configurable thresholds
- quality regression detection
- performance regression detection
- quality gates

**Status: Substantially implemented**

---

# Design Goals

The project prioritizes:

### Reproducibility

Evaluation definitions and experiment identities should be deterministic and inspectable.

### Extensibility

New evaluators and AI workflow components should be possible without redesigning the core engine.

### Observability

A failed evaluation should provide more information than simply `PASS` or `FAIL`.

### Regression Safety

Changes to AI systems should be measurable against a baseline before being accepted.

### CI Compatibility

Evaluation and regression results should be usable from automated development workflows.

### Engineering Quality

The implementation is developed incrementally using tests, focused changes, and version-controlled milestones.

---

# Example Applications

The engine is intended to support evaluation of systems such as:

- question-answering systems
- RAG pipelines
- retrieval systems
- LLM applications
- agent workflows
- model/prompt experiments
- AI services deployed behind APIs
- regression tests for AI behavior

It can be integrated into an existing AI application rather than requiring the application itself to be rebuilt around the engine.

---

# Project Philosophy

AI evaluation should not be treated as a final manual check.

It should become part of the engineering lifecycle:

```text
Build
  ↓
Evaluate
  ↓
Trace
  ↓
Compare
  ↓
Detect regressions
  ↓
Gate changes
  ↓
Deploy
```

The goal of `ai-eval-engine` is to provide the infrastructure for that workflow.

---

# License

License information will be added as the project is prepared for broader distribution.