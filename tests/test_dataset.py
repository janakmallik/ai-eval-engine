from aieval.dataset import EvalCase


def test_eval_case():
    case = EvalCase(
        id="001",
        input="What is 2 + 2?",
        expected="4",
    )

    assert case.id == "001"
    assert case.input == "What is 2 + 2?"
    assert case.expected == "4"
