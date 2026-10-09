import pytest

from aieval.rag import RAGPipeline


def test_rag_pipeline_retrieves_documents_and_generates_answer():
    calls = []

    def retriever(query: str, top_k: int):
        calls.append(("retriever", query, top_k))
        return [
            "Paris is the capital of France.",
            "France is in Europe.",
        ][:top_k]

    def generator(question: str, documents: list[str]) -> str:
        calls.append(("generator", question, documents))
        return "Paris"

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        top_k=2,
    )

    result = pipeline.run("What is the capital of France?")

    assert result.answer == "Paris"
    assert result.documents == [
        "Paris is the capital of France.",
        "France is in Europe.",
    ]

    assert calls == [
        (
            "retriever",
            "What is the capital of France?",
            2,
        ),
        (
            "generator",
            "What is the capital of France?",
            [
                "Paris is the capital of France.",
                "France is in Europe.",
            ],
        ),
    ]


def test_rag_pipeline_rejects_non_positive_top_k():
    def retriever(query: str, top_k: int):
        return []

    def generator(question: str, documents: list[str]) -> str:
        return ""

    with pytest.raises(ValueError, match="top_k must be positive"):
        RAGPipeline(
            retriever=retriever,
            generator=generator,
            top_k=0,
        )


def test_rag_pipeline_propagates_retriever_exception():
    def retriever(query: str, top_k: int):
        raise RuntimeError("retrieval failed")

    def generator(question: str, documents: list[str]) -> str:
        return "unused"

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    with pytest.raises(RuntimeError, match="retrieval failed"):
        pipeline.run("What is the capital of France?")


def test_rag_pipeline_supports_reranker():
    def retriever(query: str, top_k: int):
        return ["document 1", "document 2", "document 3"]

    def reranker(query: str, documents: list[str]):
        return ["document 3", "document 1"]

    def generator(question: str, documents: list[str]) -> str:
        return documents[0]

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
    )

    result = pipeline.run("What is relevant?")

    assert result.documents == ["document 3", "document 1"]
    assert result.answer == "document 3"


def test_rag_pipeline_does_not_call_reranker_when_not_configured():
    def retriever(query: str, top_k: int):
        return ["document 1", "document 2"]

    def generator(question: str, documents: list[str]) -> str:
        return "answer"

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    result = pipeline.run("question")

    assert result.documents == ["document 1", "document 2"]
    assert result.answer == "answer"


def test_rag_result_can_be_evaluated_with_retrieval_evaluator():
    from aieval import EvalCase, EvaluationContext
    from aieval.evaluators.retrieval_recall import RetrievalRecallEvaluator

    case = EvalCase(
        id="1",
        input="What is the capital of France?",
        expected=["Paris"],
    )

    context = EvaluationContext(
        case=case,
        actual="Paris is the capital of France.",
        retrieved=["Paris", "France"],
    )

    result = RetrievalRecallEvaluator().evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_rag_pipeline_result_contains_question():
    def retriever(query: str, top_k: int):
        return ["Paris is the capital of France."]

    def generator(question: str, documents: list[str]) -> str:
        return "Paris"

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    result = pipeline.run("What is the capital of France?")

    assert result.question == "What is the capital of France?"
    assert result.documents == ["Paris is the capital of France."]
    assert result.answer == "Paris"


def test_rag_pipeline_records_trace():
    def retriever(query: str, top_k: int):
        return ["Paris is the capital of France."]

    def generator(question: str, documents: list[str]) -> str:
        return "Paris"

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    result = pipeline.run("What is the capital of France?")

    assert result.trace is not None
