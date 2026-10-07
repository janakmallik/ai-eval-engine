[1mdiff --git a/.gitignore b/.gitignore[m
[1mindex 4330a37..42e0671 100644[m
[1m--- a/.gitignore[m
[1m+++ b/.gitignore[m
[36m@@ -15,4 +15,5 @@[m [mproject_roadmap.md[m
 baseline.json[m
 current.json[m
 project_roadmap1.md[m
[31m-cleanup_tests.py[m
\ No newline at end of file[m
[32m+[m[32mcleanup_tests.py[m
[32m+[m[32mexperiments.json[m
\ No newline at end of file[m
[1mdiff --git a/src/aieval/__init__.py b/src/aieval/__init__.py[m
[1mindex a825b0d..2da69ff 100644[m
[1m--- a/src/aieval/__init__.py[m
[1m+++ b/src/aieval/__init__.py[m
[36m@@ -1,12 +1,15 @@[m
[32m+[m[32mfrom aieval.config import EvaluationConfig[m
 from aieval.context import EvaluationContext[m
 from aieval.dataset import EvalCase, EvalDataset[m
 from aieval.evaluators.contains import ContainsEvaluator[m
 from aieval.evaluators.exact_match import ExactMatchEvaluator[m
 from aieval.evaluators.length import LengthEvaluator[m
 from aieval.evaluators.similarity import SimilarityEvaluator[m
[32m+[m[32mfrom aieval.experiment import Experiment[m
 from aieval.result import EvaluationResult[m
 from aieval.run import EvaluationRun[m
 from aieval.runner import evaluate_dataset[m
[32m+[m[32mfrom aieval.store import ExperimentStore[m
 [m
 __all__ = [[m
     "EvalCase",[m
[36m@@ -19,4 +22,7 @@[m [m__all__ = [[m
     "SimilarityEvaluator",[m
     "LengthEvaluator",[m
     "evaluate_dataset",[m
[32m+[m[32m    "EvaluationConfig",[m
[32m+[m[32m    "Experiment",[m
[32m+[m[32m    "ExperimentStore",[m
 ][m
[1mdiff --git a/src/aieval/store.py b/src/aieval/store.py[m
[1mindex 816283e..e26a861 100644[m
[1m--- a/src/aieval/store.py[m
[1m+++ b/src/aieval/store.py[m
[36m@@ -5,6 +5,7 @@[m [mfrom aieval.config import EvaluationConfig[m
 from aieval.experiment import Experiment[m
 from aieval.result import EvaluationResult[m
 from aieval.run import EvaluationRun[m
[32m+[m[32mfrom aieval.tracing.trace import Trace[m
 [m
 [m
 class ExperimentStore:[m
[36m@@ -106,7 +107,11 @@[m [mclass ExperimentStore:[m
         run = EvaluationRun([m
             results=results,[m
             metadata=run_data.get("metadata", {}),[m
[31m-            trace=None,[m
[32m+[m[32m            trace=([m
[32m+[m[32m                Trace.from_dict(run_data["trace"])[m
[32m+[m[32m                if run_data.get("trace") is not None[m
[32m+[m[32m                else None[m
[32m+[m[32m            ),[m
         )[m
 [m
         return Experiment([m
[1mdiff --git a/tests/test_experiment_store.py b/tests/test_experiment_store.py[m
[1mindex 1caa45e..51ca366 100644[m
[1m--- a/tests/test_experiment_store.py[m
[1m+++ b/tests/test_experiment_store.py[m
[36m@@ -414,3 +414,22 @@[m [mdef test_experiment_store_delete_does_not_modify_store_when_missing(tmp_path):[m
 [m
     assert deleted is False[m
     assert after == before[m
[32m+[m
[32m+[m
[32m+[m[32mdef test_experiment_store_preserves_trace(tmp_path):[m
[32m+[m[32m    from aieval.tracing.trace import Trace[m
[32m+[m
[32m+[m[32m    store = ExperimentStore(tmp_path / "experiments.json")[m
[32m+[m
[32m+[m[32m    trace = Trace()[m
[32m+[m[32m    with trace.start_span("evaluation"):[m
[32m+[m[32m        pass[m
[32m+[m
[32m+[m[32m    experiment = create_experiment()[m
[32m+[m[32m    experiment.run.trace = trace[m
[32m+[m
[32m+[m[32m    store.save(experiment)[m
[32m+[m
[32m+[m[32m    loaded = store.load(experiment.experiment_id)[m
[32m+[m
[32m+[m[32m    assert loaded.run.trace is not None[m
