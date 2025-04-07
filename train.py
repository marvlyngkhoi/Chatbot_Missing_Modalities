# from utils import *
# from trainer import *

# set_seed()


# #model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

# model_id = "google/gemma-3-4b-it"

# trainer = CustomTrainer(model_path=model_id)
# trainer.train()

import sys
from utils import *
from trainer import *

def set_model(model_choice):
    if model_choice == "1":
        return "Qwen/Qwen2.5-VL-3B-Instruct"
    elif model_choice == "2":
        return "google/gemma-3-4b-it"
    elif model_choice=='3':
        return  "HuggingFaceTB/SmolVLM-Instruct"
    elif model_choice=='4':
        return  "meta-llama/Llama-3.2-11B-Vision-Instruct"
        
    else:
        raise ValueError("Invalid choice. Use '1' for Qwen or '2' for Gemma.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_model.py <model_choice>")
        sys.exit(1)
    
    model_choice = sys.argv[1]
    model_id = set_model(model_choice)
    
    set_seed()
    trainer = CustomTrainer(model_path=model_id)
    trainer.train()
