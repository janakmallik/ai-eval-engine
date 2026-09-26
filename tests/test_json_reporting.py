import json

from aieval.reporting.json import JsonReporter
from aieval.result import EvaluationResult
from aieval.run import EvaluationRun


def test_json_reporter_renders_evaluation_run():
    results = [
        EvaluationResult(
            case_id="001",
            evaluator_name="exact_match",
            expected="Paris",
            actual="Paris",
            score=1.0,
            passed=True,
        ),
        EvaluationResult(
            case_id="002",
            evaluator_name="exact_match",
            expected="London",
            actual="Paris",
            score=0.0,
            passed=False,
        ),
    ]

    run = EvaluationRun(results)

    reporter = JsonReporter()

    output = reporter.render(run)

    data = json.loads(output)

    assert data["total"] == 2
    assert data["passed"] == 1
    assert data["failed"] == 1
    assert data["score"] == 0.5
    assert data["pass_rate"] == 0.5

    assert data["results"] == [
        {
            "case_id": "001",
            "evaluator_name": "exact_match",
            "expected": "Paris",
            "actual": "Paris",
            "score": 1.0,
            "passed": True,
        },
        {
            "case_id": "002",
            "evaluator_name": "exact_match",
            "expected": "London",
            "actual": "Paris",
            "score": 0.0,
            "passed": False,
        },
    ]

def test_json_reporter_writes_file(tmp_path):
    results = [
        EvaluationResult(
            case_id="001",
            evaluator_name="exact_match",
            expected="4",
            actual="4",
            score=1.0,
            passed=True,
        )
    ]

    run = EvaluationRun(results)

    output_path = tmp_path / "evaluation_report.json"

    reporter = JsonReporter()
    reporter.write(run, output_path)

    assert output_path.exists()

    assert json.loads(output_path.read_text()) == {
        "results": [
            {
                "case_id": "001",
                "evaluator_name": "exact_match",
                "expected": "4",
                "actual": "4",
                "score": 1.0,
                "passed": True,
            }
        ],
        "total": 1,
        "passed": 1,
        "failed": 0,
        "score": 1.0,
        "pass_rate": 1.0,
    }