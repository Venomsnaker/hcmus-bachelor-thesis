from transformers import AutoModelForCausalLM, AutoTokenizer

class FactScore:
    def __init__(
        self,
        model_name:str = "Qwen/Qwen3-4B-Instruct-2507",
        retrieve_user_prompt_path: str = "",
        retrieve_system_prompt_paht: str = ""
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )
        with open(retrieve_system_prompt_paht) as f:
            self.retrieve_system_prompt_template = f.read()
        with open(retrieve_user_prompt_path) as f:
            self.retrieve_user_prompt_template = f.read()
        
    def _generate(self, prompt: str, max_tokens=128):
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([inputs], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_tokens)
        outputs_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = self.tokenizer.decode(outputs_ids, skip_special_tokens=True)
        return content
    
    def _postprocess_facts(facts: str):
        res = []
        
        for fact in facts.split('\n'):
            fact = fact.strip()

            if fact:
                sub_facts = [f.strip() for f in fact.split('.') if f.strip()] 
                res.extend(sub_facts)
        return res
        
    
    def _retrieve_facts(self, response: str):
        prompt = self.retrieve_user_prompt_template.format(response=response)
        facts = self._generate(response, 4)
        return self._postprocess_facts(facts)
        