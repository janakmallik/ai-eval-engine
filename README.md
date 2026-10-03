## V3 - AI Tracing & Observability

V3 extends the evaluation engine with a lightweight tracing system for understanding **how an AI evaluation run executes and where failures occur**.

The goal is to make evaluation results more useful for debugging by connecting evaluation outcomes with the execution path that produced them.

### Tracing Architecture

```text
Evaluation Run
      |
      v
    Trace
      |
      +-- Evaluation
      |      |
      |      +-- Evaluation Case
      |      |      +-- Model Call
      |      |
      |      +-- Evaluator
      |             +-- Evaluation Result
      |
      +-- ...
```

### Current Capabilities

* Trace and span data structures
* Unique trace and span identifiers
* Parent/child span relationships
* Nested execution spans
* Start and end timestamps
* Duration measurement
* Span attributes and metadata
* Span lifecycle management
* Context-manager based instrumentation
* Exception recording
* Error status tracking
* Model-call tracing
* Evaluation-case tracing
* Evaluator tracing
* Trace serialization to JSON
* Trace restoration from serialized reports
* Traces attached directly to `EvaluationRun`
* JSON reporting with trace data

### Example

A traced evaluation run can represent execution such as:

```text
evaluation
│
├── evaluation.case
│   │
│   ├── model
│   │   └── 812 ms
│   │
│   ├── evaluator.exact_match
│   │   └── 0.2 ms
│   │
│   └── evaluator.similarity
│       └── 1.4 ms
│
└── evaluation.case
    └── ...
```

When an operation fails, the corresponding span can capture the failure information, including status, exception details, and execution timing.

### Design Goals

The tracing layer is intentionally implemented as a small, understandable system rather than depending directly on a full observability framework.

The project is designed to explore the engineering concepts behind production AI observability:

* How traces and spans are structured
* How nested operations are represented
* How execution timing is measured
* How failures are associated with operations
* How tracing data can be persisted
* How evaluation results and execution traces can be correlated

### Testing

Tracing is developed using a TDD workflow with dedicated tests covering:

* Trace creation
* Span creation
* Parent/child relationships
* Span lifecycle
* Duration measurement
* Attributes
* Events
* Exception recording
* Serialization
* Deserialization
* Nested traces
* Evaluation-run integration
* JSON reporting

### Next Steps

V3 will continue toward richer AI application observability, including:

* Failure-aware model and evaluator tracing
* Retrieval tracing
* Tool-call tracing
* Improved instrumentation APIs
* More structured execution metadata

These capabilities will provide the foundation for later stages of the project, including **latency, token, cost, regression, RAG, and agent evaluation**.
