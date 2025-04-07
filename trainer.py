from configs import *
from dataloader import *
from model_loader import *
import torch
import random
from trl import SFTTrainer
from huggingface_hub import login


class CustomTrainer:
    def __init__(self,
                 model_path='Qwen/Qwen2-VL-2B-Instruct',
                 dataset='zera09/finance_chat_v2',
                 configs = None,
                 output_dir='',
                 device='cuda'):
        loader = Loader()

    
        self.processor,self.model,self.model_id = loader.load(model_name=model_path)
        self.model.to(device)

        train_data,eval_data = load_and_process(data_path=dataset)

        if not configs:
            self.training_args = get_configs(output_dir=self.model_id+output_dir)
        
        self.lora_configs = get_peft_configs()
        
        collate = Collator(self.model,self.processor)
        
        if self.model_id in ['gemma']:
            self.collate_fn = collate.collate_fn_auto

        self.collate_fn = collate.collate_fn
        self.trainer = SFTTrainer(
            model=self.model,
            args = self.training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            peft_config=self.lora_configs,
            data_collator=self.collate_fn,
            tokenizer = self.processor.tokenizer,
        )
    
    def train(self,push_to_hub=True,api=None):
        self.trainer.train()

        if push_to_hub:
            login('')
            self.trainer.push_to_hub()

        
        






