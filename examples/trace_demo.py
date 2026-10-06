from aieval.dataset import EvalCase
from aieval.evaluators.exact_match import ExactMatchEvaluator
from aieval.reporting.json import JsonReporter
from aieval.runner import evaluate_dataset

dataset = [
    EvalCase(
        id="1",
        input="What is the capital of France?",
        expected="Paris",
    ),
    EvalCase(
        id="2",
        input="What is the capital of Italy?",
        expected="Rome",
    ),
]


def model(prompt):
    if "What is the capital of France?" in prompt:
        return "Paris"

    if "What is the capital of Italy?" in prompt:
        return "Rome"

    raise ValueError(f"Unknown prompt: {prompt}")


def retriever(query, top_k):
    return [
        f"Document about {query}",
        f"Another document about {query}",
    ][:top_k]


def tool(query):
    return [
        f"Tool result for {query}",
        "Additional tool result",
    ]


run = evaluate_dataset(
    model=model,
    dataset=dataset,
    evaluators=[ExactMatchEvaluator()],
    metadata={
        "model": "demo-model",
        "dataset": "capitals-v1",
    },
    retriever=retriever,
    retrieval_top_k=2,
    tool=tool,
    enable_tracing=True,
)

JsonReporter().write(run, "trace_demo.json")

print("Created trace_demo.json")
