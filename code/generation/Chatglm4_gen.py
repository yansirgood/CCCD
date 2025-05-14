import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from generation.model_gen import model_generator
import re
import random

class Chatglm4_generator(model_generator):
    def __init__(self, f_path, is_few_shot, device, is_vllm, model_path, model_name, tensor_parallel_size, gpu_memory_utilization):
        '''
        Args:
            model_name: The name of the model
        '''
        super(Chatglm4_generator, self).__init__(f_path, is_few_shot, device, is_vllm, model_path, tensor_parallel_size, gpu_memory_utilization)
        self.model_name = model_name
    def model_init(self):
        '''
        Init the model
        '''
        print(f"Initializing model in {'vLLM' if self.is_vllm else 'transformers'} mode...")
        if self.is_vllm:
            model, tokenizer = self.model_init_vllm()
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).half().cuda()
            model = model.eval()
        print("Model initialization completed.")
        return model, tokenizer

    def generate_output(self, q_type, tokenizer, model, batch_size=None):
        task_name = self.f_path.split("/")[-1].split(".")[0]
        instruction_ls, input_text_ls, answer_ls = self.process_prompt(task_name)
        output_ls = []
        num_unsuccess = 0

        for instruction, input_text, answer in tqdm(zip(instruction_ls, input_text_ls, answer_ls),
                                                    total=len(input_text_ls)):

            prompt =instruction + input_text
            prompt = model_generator.truncate_long(prompt, 4096, tokenizer, q_type)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            try:
                if q_type == "multiple_choice":
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=20,
                        do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        temperature=0
                    )
                else:
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=512,
                        do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        temperature=0
                    )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                assistant_start = response.find("<|assistant|>") + len("<|assistant|>")
                response = response[assistant_start:].strip()


                if q_type == "multiple_choice":
                    match = re.search(r'\b([A-D])\b', response.upper())
                    if match:
                        answer_pred = match.group(1)
                    else:
                        answer_pred = response
                else:
                    answer_pred = response.strip()
            except Exception as e:
                num_unsuccess += 1
                print(f"Fail to answer: {e}")
                answer_pred = "未成功回答"

            output_ls.append({"input": input_text, "output": answer_pred, "answer": answer})

        return output_ls, num_unsuccess