import torch
import json
import logging
import random
from typing import Union, Dict, Optional, List
from tqdm import tqdm
import gc

from .candidate_model_inference import CandidateBatchInferenceProvider
from .pandalm_inference import PandaLMBatchInferenceProvider


class EvaluationPipeline:
    def __init__(
        self,
        candidate_paths: List[str],
        pandalm_path: str = "WeOpenML/PandaLM-7B-v1",
        input_data_path: Optional[str] = None,
        output_data_path: Optional[str] = None,
        seed: Optional[int] = 2023,
        log_level: Optional[int] = logging.INFO,
    ):
        logging.basicConfig(
            format="%(asctime)s %(levelname)s %(message)s", level=log_level
        )
        self.input_data_path = input_data_path
        self.output_data_path = output_data_path
        self.pandalm_path = pandalm_path
        self.seed = seed

        if len(candidate_paths) < 2:
            raise ValueError(
                f"At least two candidate models are required, provided candidate_paths: {candidate_paths}."
            )

        if self.input_data_path:
            with open(input_data_path) as f:
                self.input_data = json.load(f)
        else:
            logging.info(f"No input_data_path provided, skipping candidate inference")
            for candidate in candidate_paths:
                if not candidate.endswith(".json"):
                    raise ValueError(
                        f"Candidate inference skipped, please pass .json inference results instead of {candidate}"
                    )

        self.candidate_paths = candidate_paths
        self.candidate_results = {}
        self.pandalm_results = {}
        self.pandalm_results_parsed = {}

    def seed_everything(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

    def inference_candidate(self, candidate_path):
        candidate = CandidateBatchInferenceProvider(candidate_path)
        for item in self.input_data:
            candidate.preprocess_input(
                instruction=item["instruction"],
                input=item["input"],
            )
        logging.info(f"Running inference on candidate model: {candidate_path}")
        generated = candidate.inference().copy()
        del candidate
        gc.collect()
        torch.cuda.empty_cache()
        return generated

    def collect_all_candidates(self):
        """
        Run inference on all candidate models.
        """
        for candidate_path in self.candidate_paths:
            if candidate_path.endswith(".json"):
                logging.info(
                    f"Loading candidate inference result and skipping inference:{candidate_path}"
                )
                with open(candidate_path) as f:
                    self.candidate_results[candidate_path] = json.load(f)
            else:
                logging.info(
                    f"Loading candidate model and inferencing: {candidate_path}"
                )
                generated = self.inference_candidate(candidate_path)
                self.candidate_results[candidate_path] = generated

    def pandalm_inference(self):
        """
        Run inference on the PandaLM model.
        """
        logging.info(f"Loading PandaLM model: {self.pandalm_path}")
        pandalm = PandaLMBatchInferenceProvider(self.pandalm_path)

        for i in range(len(self.candidate_paths)):
            for j in range(i + 1, len(self.candidate_paths)):
                pandalm.prepared = []
                candidate1 = self.candidate_paths[i]
                candidate2 = self.candidate_paths[j]
                logging.info(
                    f"Running inference on PandaLM model with candidate1:{candidate1}, candidate2:{candidate2}"
                )
                cand1_results = self.candidate_results[candidate1]
                cand2_results = self.candidate_results[candidate2]

                assert len(cand1_results) == len(cand2_results)

                for idx in range(len(cand1_results)):
                    pandalm.preprocess_input(
                        instruction=self.input_data[idx]["instruction"],
                        input=self.input_data[idx]["input"],
                        response1=cand1_results[idx],
                        response2=cand2_results[idx],
                    )
                generated = pandalm.inference().copy()

                self.pandalm_results[(candidate1, candidate2)] = generated
                parsed = []
                for item in generated:
                    parsed.append(pandalm.parse_pandalm_response(item))
                self.pandalm_results_parsed[(candidate1, candidate2)] = parsed
        del pandalm
        gc.collect()
        torch.cuda.empty_cache()
        return self.pandalm_results_parsed

    def evaluate(self):
        self.seed_everything(self.seed)
        self.collect_all_candidates()
        parsed_results = self.pandalm_inference()
        return parsed_results
