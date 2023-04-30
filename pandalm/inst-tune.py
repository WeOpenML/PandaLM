from dataclasses import dataclass, field
from typing import Optional
import transformers

# import core classes
from core import CustomTrainer, Nets, Datasets, DataCollatorForDataset
# import global variables
from core.global_var import IGNORE_INDEX, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, PROMPT_DICT 

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")

@dataclass
class PeftArguments:
    peft_model: Optional[str] = field(default="lora")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    deepspeed: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def train():
    parser = transformers.HfArgumentParser((ModelArguments, PeftArguments, DataArguments, TrainingArguments))
    model_args, peft_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    #prepare model and tokenizer for training
    nets = Nets(model_args, training_args, peft_args)
    model = nets.get_model()
    tokenizer = nets.get_tokenizer()
    
 
    #prepare datasets for training and validation
    train_dataset = Datasets(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForDataset(tokenizer=tokenizer)
     
    # prepare trainer and train model
    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
    trainer.train()
    trainer.save_state()
    
    # save model after train
    nets.safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
