from aieval import (
    AnswerRelevanceEvaluator,
    ContextRelevanceEvaluator,
    EvalCase,
    FaithfulnessEvaluator,
    RAGPipeline,
)
from aieval.context import EvaluationContext


def retrieve(question: str, top_k: int) -> list[str]:
    documents = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
    ]
    return documents[:top_k]


def generate_answer(question: str, documents: list[str]) -> str:
    return "Paris is the capital of France."


def run_demo():
    pipeline = RAGPipeline(
        retriever=retrieve,
        generator=generate_answer,
        top_k=2,
    )

    rag_result = pipeline.run("What is the capital of France?")

    case = EvalCase(
        id="rag-001",
        input=rag_result.question,
        expected="Paris is the capital of France.",
    )

    context = EvaluationContext(
        case=case,
        actual=rag_result.answer,
        retrieved=rag_result.documents,
    )

    evaluators = [
        ContextRelevanceEvaluator(),
        AnswerRelevanceEvaluator(),
        FaithfulnessEvaluator(),
    ]

    results = [evaluator.evaluate(context) for evaluator in evaluators]

    return rag_result, results


def main() -> None:
    rag_result, results = run_demo()

    print("RAG Evaluation Demo")
    print("===================")
    print(f"Question: {rag_result.question}")

    print("\nRetrieved documents:")
    for document in rag_result.documents:
        print(f"- {document}")

    print(f"\nAnswer: {rag_result.answer}")

    print("\nEvaluation results:")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"- {result.evaluator_name}: score={result.score:.2f}, status={status}")

    print("\nTrace:")
    print(rag_result.trace.to_dict())


if __name__ == "__main__":
    main()
