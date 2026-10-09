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
        with Trace() as trace:
            with trace.start_retrieval(
                query=question,
                top_k=self.top_k,
            ) as retrieval_span:
                documents = self.retriever(question, self.top_k)

                retrieval_span.set_attribute(
                    "retrieval.result_count",
                    len(documents),
                )

            if self.reranker is not None:
                documents = self.reranker(question, documents)

            with trace.start_span("generation") as generation_span:
                answer = self.generator(question, documents)
                generation_span.set_attribute(
                    "generation.answer_length",
                    len(answer),
                )

            return RAGResult(
                question=question,
                answer=answer,
                documents=documents,
                trace=trace,
            )
