import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

import io
from PIL import Image
import torch
from tqdm import tqdm
from datasets import load_dataset
import json

from transformers import AutoProcessor, Gemma3ForConditionalGeneration


def process_multimodal_dataset(model, processor, dataset, max_new_tokens=1024, device="cuda", output_file="results_gemma.json"):
    results = {}

    
    
    for i in tqdm(range(len(dataset['train']['conversation']))):
        conversation_id = dataset['train']['conversation_id'][i]
        user_msgs, assistant_msgs = [], []
        
        # Extract user and assistant messages
        for conv in dataset['train']['conversation'][i]:
            if conv[0]['role'] == 'user':
                user_msgs.append(conv[0])
            else:
                assistant_msgs.append(conv[0])
        
        # Process image
        image_data = dataset['train']['images'][i][0]['bytes']
        image = Image.open(io.BytesIO(image_data)).resize((200, 200))
        
        messages_img = []
        messages_no_img = []
        label = None
        
        for idx in range(len(assistant_msgs)):
            if idx==0:
                messages_img.append({
                    'role': 'user',
                    'content': [
                        {"type": "image", "image": image},
                        {'type': 'text', 'text': user_msgs[idx]['content']}]
                })
            else:
                messages_img.append({
                    'role': 'user',
                    'content': [{'type': 'text', 'text': user_msgs[idx]['content']}]
                })
            messages_no_img.append({
                'role': 'user',
                'content': [{'type': 'text', 'text': user_msgs[idx]['content']}]
            })
            
            if len(assistant_msgs) - 1 > idx:
                messages_img.append({
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': assistant_msgs[idx]['content']}]
                })
                messages_no_img.append({
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': assistant_msgs[idx]['content']}]
                })
            else:
                label = assistant_msgs[idx]
        
        
        
        # Processing with image
        inputs_img = processor.apply_chat_template(
            messages_img, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        
        input_len_img = inputs_img["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation_img = model.generate(**inputs_img, max_new_tokens=max_new_tokens, do_sample=False)
            generation_img = generation_img[0][input_len_img:]
        
        output_text_img = processor.decode(generation_img, skip_special_tokens=True)
        
        # Processing without image
        inputs_no_img = processor.apply_chat_template(
            messages_no_img, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        input_len_no_img = inputs_no_img["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation_no_img = model.generate(**inputs_no_img, max_new_tokens=max_new_tokens,do_sample=False)
            generation_no_img = generation_no_img[0][input_len_no_img:]

        output_text_no_img = processor.decode(generation_no_img, skip_special_tokens=True)
        
        results[conversation_id] = {
            "output_with_image": output_text_img,
            "output_without_image": output_text_no_img,
            "label": label
        }
    # Write results to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    #return results, output_file



# Get dataset 
ds = load_dataset("zera09/sampled_visionarena_chat_100")

model_id = "google/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)



process_multimodal_dataset(model, processor, ds, max_new_tokens=512)
