from transformers import Trainer
import logging
class CustomTrainer(Trainer):
    def placeholder_func(self):
        logging.warning("Using CustomTrainer. This is a placeholder fuction...")
 
