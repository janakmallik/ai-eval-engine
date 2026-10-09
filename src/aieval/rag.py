from collections.abc import Callable
from dataclasses import dataclass

from aieval.tracing.trace import Trace


@dataclass(frozen=True)
class RAGResult:
    question: str
    answer: str
    documents: list[str]
    trace: Trace


class RAGPipeline:
    def __init__(
        self,
        retriever: Callable[[str, int], list[str]],
        generator: Callable[[str, list[str]], str],
        top_k: int = 5,
        reranker: Callable[[str, list[str]], list[str]] | None = None,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.top_k = top_k

    def run(self, question: str) -> RAGResult:
        trace = Trace()

        retrieval_span = trace.start_retrieval(
            query=question,
            top_k=self.top_k,
        )

        documents = self.retriever(question, self.top_k)

        retrieval_span.set_attribute(
            "retrieval.result_count",
            len(documents),
        )

        if self.reranker is not None:
            documents = self.reranker(question, documents)

        answer = self.generator(question, documents)

        return RAGResult(
            question=question,
            answer=answer,
            documents=documents,
            trace=trace,
        )
