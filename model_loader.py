from transformers import AutoProcessor
import torch

from transformers import (
    Gemma3ForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    MllamaForConditionalGeneration,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq
)



model_map = {
    'gemma':Gemma3ForConditionalGeneration,
    'Qwen2.5':Qwen2_5_VLForConditionalGeneration,
    'Llama': MllamaForConditionalGeneration,
    'SmolVLM':AutoModelForVision2Seq
}



class Loader:
    def __init__(self):
        pass
    def  load(self,model_name = 'google/gemma-3-27b-it'):
        model_id = model_name.split('/')[1].split('-')[0]
        
        print(model_id,model_name)

        processor = AutoProcessor.from_pretrained(model_name)
        model = model_map[model_id].from_pretrained(model_name)

        return processor, model, model_id 
