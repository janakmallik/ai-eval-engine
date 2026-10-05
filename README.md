the product layer
```
                    CURRENT ENGINE
                         │
        ┌────────────────┼────────────────┐
        │                │                │
       API              CLI             JSON
        │                │                │
        └────────────────┼────────────────┘
                         │
                    CI / Quality Gate
                         │
                    GitHub Actions
                         │
                    Documentation
                         │
                       DONE
```

current architecture is roughly:

```text
                    ai-eval-engine
                           │
          ┌────────────────┼────────────────┐
          │                │                │
      Python API       Evaluators        Tracing
          │                │                │
      runner.py       exact/contains     model
          │            similarity        retrieval
          │            LLM judge         tool
          │            RAG metrics       usage
          │
          ├───────────────┐
          │               │
      EvaluationRun   EvaluationResult
          │
          ├───────────────┐
          │               │
      JSON Reporter    Regression
          │               │
      baseline.json    comparison
      current.json     detector
                          │
                       Gate
                          │
                        CLI
                          │
                    GitHub Actions
```

That's a good foundation.

But I see **three different layers that we need to finish separately**:

### 1. Core engine

Mostly done.

```text
Python API
Evaluators
RAG evaluation
Tracing
Tool tracing
Token usage
Cost/usage data
Experiments
Regression detection
Quality gates
```

### 2. Developer interface

Partially done.

```text
CLI
JSON reports
pip install
CI
GitHub Actions
```

Your CLI currently has:

```bash
aieval regression \
    --baseline baseline.json \
    --current current.json
```

That's useful, but we need to determine whether that's the **final CLI surface** or merely the first command.

### 3. Actual user experience

This is where we're currently weakest.

Someone installing your package should be able to understand:

```text
pip install ai-eval-engine
```

then:

```python
from aieval.runner import evaluate_dataset
```

then:

```text
run evaluation
      ↓
save JSON
      ↓
compare against baseline
      ↓
run quality gate
      ↓
exit 0 / exit 1
```

without having to inspect your GitHub source code.

That's what the final `.md` instructions and README need to solve.

---

<!-- ## Before we touch more code, I want 5 more things

You don't need to give me the README.

Please give me these:

```powershell
Get-Content src\aieval\run.py
```

```powershell
Get-Content src\aieval\summary.py
```

```powershell
Get-Content src\aieval\experiment.py
```

```powershell
Get-Content src\aieval\__init__.py
```

And:

```powershell
Get-ChildItem tests -File | Select-Object Name
```

I particularly want to see the tests because **the tests will tell me what functionality you already consider part of the public contract**.

### One thing I already noticed

Your `JsonReporter` is currently doing this:

```python
return EvaluationRun(
    results=results,
    metadata=data.get("metadata", {}),
    trace=(
        Trace.from_dict(data["trace"])
        if data.get("trace") is not None
        else None
    ),
)
```

So JSON persistence already works at the reporter level.

That's good.

But earlier we encountered:

```text
EvaluationRun.from_dict(...)
AttributeError
```

So we should decide whether deserialization belongs to:

```text
JsonReporter.read()
```

or:

```text
EvaluationRun.from_dict()
```

rather than accidentally maintaining two competing serialization APIs.

That's exactly the sort of architectural cleanup I want to catch **before** we declare the project finished.

---

## And I want to preserve your current stopping rule

I won't keep adding features indefinitely.

Our eventual definition of **DONE** will be:

```text
Python API
    ↓
CLI
    ↓
JSON Reports
    ↓
CI Quality Gate
    ↓
pip install
    ↓
Real project integration
    ↓
README + examples
    ↓
Clear .md instructions
    ↓
GitHub Actions verified
    ↓
DONE
```

At that point, **we stop**.

Not:

> "Maybe we should add dashboards."

Not:

> "Maybe we should add distributed tracing."

Not:

> "Maybe we should add a vector database."

Not:

> "Maybe we should build an experiment server."

Those can be future projects.

Your current project should end as a **small, coherent, installable AI evaluation + observability engine that a real ML/RAG developer can actually integrate**.

Send me those 5 outputs, and I'll give you the **remaining implementation map**, including exactly what we should finish in V4, what belongs in V5, what is already sufficient, and what we should *not* build. -->