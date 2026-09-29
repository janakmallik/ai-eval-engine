import json

from aieval.dataset import EvalCase, EvalDataset


def test_eval_dataset_loads_from_json(tmp_path):
    data = [
        {
            "id": "001",
            "input": "What is 2 + 2?",
            "expected": "4",
        },
        {
            "id": "002",
            "input": "What is the capital of France?",
            "expected": "Paris",
        },
    ]

    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    dataset = EvalDataset.from_json(path)

    assert len(dataset.cases) == 2

    assert dataset.cases[0].id == "001"
    assert dataset.cases[0].input == "What is 2 + 2?"
    assert dataset.cases[0].expected == "4"

    assert dataset.cases[1].id == "002"
    assert dataset.cases[1].input == "What is the capital of France?"
    assert dataset.cases[1].expected == "Paris"

def test_eval_dataset_rejects_invalid_top_level_json(tmp_path):
    path = tmp_path / "dataset.json"

    path.write_text(
        '{"id": "001", "input": "What is 2 + 2?", "expected": "4"}',
        encoding="utf-8",
    )

    try:
        EvalDataset.from_json(path)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

def test_eval_dataset_rejects_missing_fields(tmp_path):
    data = [
        {
            "id": "001",
            "input": "What is 2 + 2?",
        }
    ]

    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    try:
        EvalDataset.from_json(path)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

def test_eval_dataset_rejects_invalid_json(tmp_path):
    path = tmp_path / "dataset.json"

    path.write_text(
        '[{"id": "001", "input": "What is 2 + 2?", "expected": "4"',
        encoding="utf-8",
    )

    try:
        EvalDataset.from_json(path)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

def test_eval_dataset_to_json(tmp_path):
    dataset = EvalDataset(
        cases=[
            EvalCase(
                id="001",
                input="What is 2 + 2?",
                expected="4",
            ),
            EvalCase(
                id="002",
                input="What is 3 + 3?",
                expected="6",
            ),
        ]
    )

    path = tmp_path / "dataset.json"

    dataset.to_json(path)

    assert json.loads(path.read_text(encoding="utf-8")) == [
        {
            "id": "001",
            "input": "What is 2 + 2?",
            "expected": "4",
        },
        {
            "id": "002",
            "input": "What is 3 + 3?",
            "expected": "6",
        },
    ]

def test_eval_dataset_is_iterable():
    dataset = EvalDataset(
        cases=[
            EvalCase(
                id="001",
                input="What is 2 + 2?",
                expected="4",
            ),
            EvalCase(
                id="002",
                input="What is 3 + 3?",
                expected="6",
            ),
        ]
    )

    cases = list(dataset)

    assert cases == dataset.cases

def test_eval_dataset_has_length():
    dataset = EvalDataset(
        cases=[
            EvalCase(
                id="001",
                input="What is 2 + 2?",
                expected="4",
            ),
            EvalCase(
                id="002",
                input="What is 3 + 3?",
                expected="6",
            ),
        ]
    )

    assert len(dataset) == 2


def test_eval_dataset_supports_indexing():
    dataset = EvalDataset(
        cases=[
            EvalCase(
                id="001",
                input="What is 2 + 2?",
                expected="4",
            ),
            EvalCase(
                id="002",
                input="What is 3 + 3?",
                expected="6",
            ),
        ]
    )

    assert dataset[0].id == "001"
    assert dataset[1].id == "002"