import transformers
from typing import  Dict
from peft import TaskType, LoraConfig, get_peft_model, prepare_model_for_int8_training,get_peft_model_state_dict
import logging

# import global variables
from llm.core.global_var import IGNORE_INDEX, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, PROMPT_DICT 


class Nets:
    def __init__(self, model_args, training_args, peft_args, **kwargs):
        super(Nets, self).__init__()

        # init model
        if 'glm' in model_args.model_name_or_path:
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                            model_args.model_name_or_path,
                            cache_dir=training_args.cache_dir,
                            trust_remote_code=True,
                        )
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                            model_args.model_name_or_path,
                            cache_dir=training_args.cache_dir,
                        )
        model = self.get_peft_model(peft_args,model) 
        # init tokenizer
        if 'llama' in model_args.model_name_or_path:
            tokenizer = transformers.LlamaTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
        elif 'glm' in model_args.model_name_or_path:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
                trust_remote_code=True,
            )
        elif 'pythia' in model_args.model_name_or_path:
            tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
        
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
 
        if tokenizer.pad_token is None:
            self.smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens(
                {
                    "eos_token": DEFAULT_EOS_TOKEN,
                    "bos_token": DEFAULT_BOS_TOKEN,
                    "unk_token": DEFAULT_UNK_TOKEN,
                }
            )

        # finish initilization
        self.model = model
        self.tokenizer = tokenizer

    def get_model(self):
        return self.model
    def get_tokenizer(self):
        return self.tokenizer
    def get_peft_model(self,peft_args,model):
        if peft_args.peft_model == 'none':
            logging.warning("Full finetuning...")
            return model
        elif peft_args.peft_model == 'lora':
            logging.warning("Using lora...")
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj","v_proj"],
                task_type=TaskType.CAUSAL_LM,
            )


        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model

 
    def smart_tokenizer_and_embedding_resize(
        self,
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
    ):
        """Resize tokenizer and embedding.

        Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
        """
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

    def safe_save_model_for_hf_trainer(self, trainer: transformers.Trainer, output_dir: str):
        """Collects the state dict and dump to disk."""
        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
