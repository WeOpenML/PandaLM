from pandalm import EvaluationPipeline

pipeline = EvaluationPipeline(
    candidate_paths=["./llama-7b-tuned/", "./bloom-7b-tuned/", "./opt-7b-tuned"],
    input_data_path="./data/pipeline-sanity-check.json",
)
print(pipeline.evaluate())
