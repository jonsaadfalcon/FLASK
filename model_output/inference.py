import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from tqdm import tqdm
#import ray
from load_model import get_conversation_template

from together import Together

import transformers
from transformers import AutoTokenizer, GenerationConfig

##################################################

def generate_candidates_with_together_api(instruction:str, 
                                          model: str, 
                                          temperature: float,
                                          previous_turns: dict = None,
                                          system_prompt: str = None):
    
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

    if system_prompt is None:
        system_prompt = "You are an expert chatbot, capable of instruction-following and question-answering. You are tasked with following the given instruction for the provided input."
    
    user_prompt = instruction

    ###################################

    if previous_turns is None:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
    else:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": previous_turns["first_instruction"]},
                    {"role": "assistant", "content": previous_turns["system_response"]},
                    {"role": "user", "content": user_prompt}]
    
    #print("-----------------------------------")
    #print("Messages: ")
    #for message in messages:
    #    print(message)
    #print("-----------------------------------")

    response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                #top_p=generation_dict['top_p'],
                #top_k=generation_dict['top_k'],
            )

    output = response.choices[0].message.content

    return output

##################################################

def load_HF_pipeline(model_path: str, max_new_tokens: int):

        model_id = model_path
        model = model_path
    
        if model == "microsoft/Phi-3-small-8k-instruct":
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                tokenizer=AutoTokenizer.from_pretrained(model_id, trust_remote_code=True),
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                trust_remote_code=True
            )
        else:
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                #model_kwargs={"torch_dtype": torch.bfloat16} if model == "meta-llama/Meta-Llama-3-8B-Instruct" else {"torch_dtype": "auto"},
                #model_kwargs={"torch_dtype": "auto"},
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                trust_remote_code=True
            )

        pipeline.model.config.pad_token_id = pipeline.tokenizer.eos_token_id
        pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
        if model in ["meta-llama/Meta-Llama-3-8B-Instruct", "princeton-nlp/Llama-3-Instruct-8B-SimPO", "princeton-nlp/Llama-3-Instruct-8B-IPO", 
                 "princeton-nlp/Llama-3-Instruct-8B-RDPO", "princeton-nlp/Llama-3-Instruct-8B-DPO"]:
            pipeline.tokenizer.padding_side = 'left'

        pipeline.model.config.is_encoder_decoder = False

        ########################################

        terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        ########################################

        generation_config, unused_kwargs = GenerationConfig.from_pretrained(
            model_id, 
            return_unused_kwargs=True
        )

        generation_config.batch_size = 1
        
        generation_config.max_new_tokens = max_new_tokens
        generation_config.do_sample = True
        #generation_config.temperature = temperature
        generation_config.top_p = 0.9
        generation_config.num_return_sequences = 1
        generation_config.is_encoder_decoder = False
        generation_config.eos_token_id = terminators if model in ["meta-llama/Meta-Llama-3-8B-Instruct"] else pipeline.tokenizer.eos_token_id
        if model in ["meta-llama/Meta-Llama-3-8B-Instruct", "princeton-nlp/Llama-3-Instruct-8B-SimPO", "princeton-nlp/Llama-3-Instruct-8B-IPO", 
                 "princeton-nlp/Llama-3-Instruct-8B-RDPO", "princeton-nlp/Llama-3-Instruct-8B-DPO"]:
            generation_config.pretraining_tp = 1
        
        pipeline.model.config = generation_config

        return pipeline, generation_config

##################################################

def generate_candidates_with_huggingface_locally(instruction:str, 
                                                 pipeline: transformers.pipeline,
                                                 generation_config: GenerationConfig,
                                                 previous_turns: dict = None):
    
    system_prompt = "You are an expert chatbot, capable of instruction-following and question-answering. You are tasked with following the given instruction for the provided input."
    user_prompt = instruction
            
    if previous_turns is None:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
    else:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": previous_turns["first_instruction"]},
                    {"role": "assistant", "content": previous_turns["system_response"]},
                    {"role": "user", "content": user_prompt}]
            
    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )
    prompt_length = len(prompt)

    outputs = pipeline(
        prompt,
        batch_size=generation_config.batch_size,
        generation_config=generation_config
    )

    answer = outputs[0]["generated_text"][prompt_length:]
    return answer
                                                 

##################################################

def search_string_in_jsonl(file_path, search_string):
    if not os.path.exists(file_path):
        return False
    
    found = False
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if search_string in line:
                found = True
                break
                #print(f"Found the string in line: {line.strip()}")
    #if not found:
     #   print(f"The string '{search_string}' was not found in the file.")
    return found

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def run_eval(model_path, model_id, question_file, answer_file, num_gpus, model_type, num_choices):
    # split question file into num_gpus files
    ques_jsons = []
    with open(os.path.expanduser(question_file), "r") as ques_file:
        for line in ques_file:
            ques_jsons.append(line)

    #chunk_size = len(ques_jsons) // num_gpus
    chunk_size = 50
    ans_handles = []
    for i in range(0, len(ques_jsons), chunk_size):
        question_string = f'"question_id": {i + 1}'
        if not search_string_in_jsonl(answer_file, question_string):
            print(f"Generating answers for questions {i} to {i + chunk_size}")
            #ans_handles.append(get_model_answers.remote(model_path, model_id, ques_jsons[i:i + chunk_size],
            #                                            model_type=model_type, num_choices=num_choices))
            ans_handles.append(get_model_answers(model_path, model_id, ques_jsons[i:i + chunk_size],
                                                model_type=model_type, num_choices=num_choices))

            ans_jsons = []
            for ans_handle in ans_handles:
                #ans_jsons.extend(ray.get(ans_handle))
                ans_jsons.extend(ans_handle)

            with open(os.path.expanduser(answer_file), "w") as ans_file:
                for line in ans_jsons:
                    ans_file.write(json.dumps(line) + "\n")
        else:
            print(f"Answers for questions {i} to {i + chunk_size} already exist in {answer_file}")


#@ray.remote(num_gpus=1)
#@torch.inference_mode()
def get_model_answers(model_path, model_id, question_jsons, model_type, num_choices, temperature=0.7, max_new_tokens=1024):

    if model_type == "local":

        disable_torch_init()
        model_path = os.path.expanduser(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast= False)
        model = AutoModelForCausalLM.from_pretrained(model_path,
            torch_dtype=torch.float16).cuda()

        ans_jsons = []
        for i, line in enumerate(tqdm(question_jsons)):
            ques_json = json.loads(line)
            idx = ques_json["question_id"]
            qs = ques_json["text"]
            #print("initial question", qs)
            conv = get_conversation_template(model_id)
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            inputs = tokenizer([prompt])
            output_ids = model.generate(
                torch.as_tensor(inputs.input_ids).cuda(),
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens)
            output_ids = output_ids[0][len(inputs.input_ids[0]) :]
            outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            if i % 10:
                print("initial question", qs)
                print("cleaned output",outputs)
            ans_jsons.append({"question_id": idx,
                             "text": outputs})
        return ans_jsons
    
    elif model_type == "TogetherAI":

        ans_jsons = []
        for i, line in enumerate(tqdm(question_jsons)):
            ques_json = json.loads(line)
            idx = ques_json["question_id"]
            qs = ques_json["text"]
            #print("initial question", qs)
            conv = get_conversation_template(model_id)
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            ############################

            instruction = qs
            system_prompt = conv.system
            previous_turns = {"first_instruction": conv.messages[0][1],
                              "system_response": conv.messages[1][1]}

            ############################
            
            total_candidates = []
            for _ in range(num_choices):
                output = generate_candidates_with_together_api(instruction=instruction, 
                                                               model=model_path, 
                                                               temperature=temperature,
                                                               #previous_turns=previous_turns,
                                                               previous_turns=None,
                                                               system_prompt=system_prompt)
                total_candidates.append(output)

            ############################

            output = total_candidates[0]

            #print("Cleaned Output: ", output)
            if i % 10:
                print("initial question", qs)
                print("cleaned output",outputs)
            ans_jsons.append({"question_id": idx,
                              "text": output,
                              "total_candidates": total_candidates})
            
        return ans_jsons
    
    elif model_type == "HuggingFace":

        pipeline, generation_config = load_HF_pipeline(model_path, max_new_tokens) 

        ans_jsons = []
        for i, line in enumerate(tqdm(question_jsons)):
            ques_json = json.loads(line)
            idx = ques_json["question_id"]
            qs = ques_json["text"]
            print("initial question", qs)
            conv = get_conversation_template(model_id)
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            ############################

            instruction = qs
            system_prompt = conv.system
            previous_turns = {"first_instruction": conv.messages[0][1],
                              "system_response": conv.messages[1][1]}

            ############################
            
            total_candidates = []
            for _ in range(num_choices):
                output = generate_candidates_with_huggingface_locally(instruction=instruction,
                                                                      pipeline=pipeline,
                                                                      generation_config=generation_config,
                                                                      previous_turns=None)
                                                                      #previous_turns=previous_turns)
                total_candidates.append(output)

            if i % 10 == 0:
                print(f"Instruction: {instruction}")
                print(f"Output: {output}")
                print("-----------------------------------------")

            ############################

            output = total_candidates[0]

            #print("Cleaned Output: ", output)
            #breakpoint()
            ans_jsons.append({"question_id": idx,
                              "text": output,
                              "total_candidates": total_candidates})
            
        return ans_jsons

    else:
        raise ValueError("Invalid model type! Model Type Given: ", model_type)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-id", type=str, default="alpaca")
    parser.add_argument("--question-file", type=str, default="../input_data/flask_evaluation_raw.jsonl")
    parser.add_argument("--answer-file", type=str, default="outputs/alpaca_7b.jsonl")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--model-type", type=str, default="TogetherAI")
    parser.add_argument("--num-choices", type=int, default=1)
    args = parser.parse_args()

    #ray.init()
    run_eval(args.model_path, args.model_id, args.question_file, args.answer_file, args.num_gpus, args.model_type, args.num_choices)
