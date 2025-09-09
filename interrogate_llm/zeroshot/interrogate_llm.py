import time
import os
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

class InterrogateLLM:
    def __init__(
        self,
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        reconstruct_prompt_template_path="",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )
        with open(reconstruct_prompt_template_path) as f:
            self.reconstruct_prompt_template = f.read()
        
    def _generate(self, prompt: str, max_tokens=1024):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_tokens)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return content
    
    def recontruct_prompt(self, context: str, response: str, max_tokens=1024):
        if context == '':
            context = 'None'
        
        prompt = self.reconstruct_prompt_template.format(
            context=context,
            response=response
        )
        return self._generate(prompt, max_tokens=1024)
    

