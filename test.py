from pandalm import EvaluationPipeline
pipeline = EvaluationPipeline(
    candidate_paths=["./llama-7b-tuned/", "./opt-7b-tuned", "some-other-model.json"], 
    input_data_path="data/pipeline-sanity-check.json",
)
print(pipeline.evaluate())