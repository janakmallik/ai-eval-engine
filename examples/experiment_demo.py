from aieval import (
    EvalCase,
    EvaluationConfig,
    ExactMatchEvaluator,
    Experiment,
    ExperimentStore,
    evaluate_dataset,
)

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


def model(prompt: str) -> str:
    answers = {
        "What is the capital of France?": "Paris",
        "What is the capital of Italy?": "Rome",
    }

    return answers[prompt]


config = EvaluationConfig(
    model="qa-model",
    model_version="v1",
    dataset="capitals-v1",
    evaluators=[ExactMatchEvaluator()],
)


run = evaluate_dataset(
    model=model,
    dataset=dataset,
    evaluators=config.evaluators,
    metadata=config.metadata,
)


experiment = Experiment.from_config(
    name="baseline",
    config=config,
    run=run,
)


store = ExperimentStore("experiments.json")
store.save(experiment)

loaded = store.load(experiment.experiment_id)


print("Experiment")
print("----------")
print(f"Name:         {loaded.name}")
print(f"Experiment ID: {loaded.experiment_id}")
print(f"Model:        {loaded.model}")
print(f"Version:      {loaded.model_version}")
print(f"Dataset:      {loaded.dataset}")
print(f"Score:        {loaded.run.score:.2%}")
print(f"Pass rate:    {loaded.run.pass_rate:.2%}")
print()
print("Saved and loaded successfully.")
