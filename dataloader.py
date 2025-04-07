import torch
from PIL import Image
import os
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from tqdm import tqdm

from transformers import (
    Gemma3ForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    MllamaForConditionalGeneration,
    AutoModelForImageTextToText,
    AutoProcessor,
    Idefics3ForConditionalGeneration
)


def load_img(path,h=336,w=336):

    img = Image.open(path)
    img = img.resize((h,w))

    return img

def padding():
    pass


def format_data(sample):
    try:
        path = sample['image_path']
    except:
        path = sample['image']
    if os.path.exists('/home/sarmistha/Sarmistha/ECML/'+path):

    
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": 'You are a finacnial assistant.'}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image":load_img('/home/sarmistha/Sarmistha/ECML/'+path),
                    },
                    {
                        "type": "text",
                        "text": sample["query"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample['answer']}],
            },
        ]
    
    else:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": 'You are a finacnial assistant.'}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": sample["query"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample['answer']}],
            },
        ]


class Collator:
    def __init__(self,model,processor):
        self.model = model
        self.processor = processor
    

    def collate_fn_auto(self,samples):
    
        texts = [
            self.processor.apply_chat_template(s,tokenize=False) for s in samples 
        ]

        batch = self.processor(
            text=texts,return_tensors = 'pt',padding=True
        )


        labels = batch['input_ids'].clone()

        

        if isinstance(self.model,Gemma3ForConditionalGeneration):
            image_token = [262144]
            labels[labels==self.processor.tokenizer.pad_token_id] =-100

        
        batch['labels'] = labels

        return batch

    def collate_fn(self,samples):
        if not samples:
            return {}  # Return empty dict if no samples (avoid crashing)

        texts = []
        image_inputs = []

        for s in samples:
            texts.append(self.processor.apply_chat_template(s, tokenize=False))  # Process text
            img = process_vision_info(s)  # Process image
            image_inputs.append(img[0] if img is not None else None)  # Handle None cases

        # Check if batch has mixed, all-text, or all-image samples
        has_images = [img is not None for img in image_inputs]
        has_text = any(texts)

        # Separate text-only and text-with-image inputs
        text_with_images = [texts[i] for i in range(len(texts)) if has_images[i]]
        images = [image_inputs[i] for i in range(len(image_inputs)) if has_images[i]]
        text_only = [texts[i] for i in range(len(texts)) if not has_images[i]]

        batch = {}

        # Case 1: All text-only samples
        if has_text and not any(has_images):
            batch = self.processor(text=text_only, return_tensors="pt", padding=True)
        
        # Case 2: All image-based samples (no text)
        elif not has_text and all(has_images):
            batch = self.processor(images=images, return_tensors="pt", padding=True)
        
        # Case 3: Mixed batch (some with text-only, some with images)
        elif has_text and any(has_images):
            text_only_batch = self.processor(text=text_only, return_tensors="pt", padding=True) if text_only else None
            multimodal_batch = self.processor(text=text_with_images, images=images, return_tensors="pt", padding=True) if text_with_images else None
            
            for key in ["input_ids", "attention_mask"]:
                if text_only_batch and multimodal_batch:
                    batch[key] = torch.cat([text_only_batch[key], multimodal_batch[key]], dim=0)
                elif text_only_batch:
                    batch[key] = text_only_batch[key]
                elif multimodal_batch:
                    batch[key] = multimodal_batch[key]

        # Handle edge case: Empty batch (e.g., corrupted samples)
        if not batch:
            return {}

        # Clone input_ids for labels
        if "input_ids" in batch:
            labels = batch["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

            # Handle model-specific image token masking
            if isinstance(self.model, Qwen2_5_VLForConditionalGeneration):
                image_tokens = [151652, 151653, 151655]
            elif isinstance(self.model,Idefics3ForConditionalGeneration):
                image_tokens = [49153]
            else:
                try:
                    image_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)]
                except AttributeError:
                    image_tokens = []  # Fallback if processor.image_token doesn't exist

            for image_token_id in image_tokens:
                labels[labels == image_token_id] = -100  

            batch["labels"] = labels  # Add labels to batch

        return batch


def load_and_process(data_path="zera09/finance_chat"):
    dataset = load_dataset(data_path, split = "train")
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = []
    eval_dataset = []

    for sample in tqdm(dataset['train']):        
        train_dataset.append(format_data(sample))
    
    for sample in tqdm(dataset['test']):
        eval_dataset.append(format_data(sample))


    return train_dataset,eval_dataset 

