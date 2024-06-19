import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from tqdm import tqdm
import ray
from load_model import get_conversation_template

from together import Together

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
                    {"role": "system", "content": previous_turns["system_response"]},
                    {"role": "user", "content": user_prompt}]
    
    print("-----------------------------------")
    print("Messages: ")
    for message in messages:
        print(message)
    print("-----------------------------------")

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

    chunk_size = len(ques_jsons) // num_gpus
    ans_handles = []
    for i in range(0, len(ques_jsons), chunk_size):
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


#@ray.remote(num_gpus=1)
#@torch.inference_mode()
def get_model_answers(model_path, model_id, question_jsons, model_type, num_choices, temperature=0.7):

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
            print("initial question", qs)
            conv = get_conversation_template(model_id)
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            inputs = tokenizer([prompt])
            output_ids = model.generate(
                torch.as_tensor(inputs.input_ids).cuda(),
                do_sample=True,
                temperature=0.7,
                max_new_tokens=1024)
            output_ids = output_ids[0][len(inputs.input_ids[0]) :]
            outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
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
            for i in range(num_choices):
                output = generate_candidates_with_together_api(instruction=instruction, 
                                                               model=model_path, 
                                                               temperature=temperature,
                                                               #previous_turns=previous_turns,
                                                               previous_turns=None,
                                                               system_prompt=system_prompt)
                total_candidates.append(output)

            ############################

            output = total_candidates[0]

            print("Cleaned Output: ", output)
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
