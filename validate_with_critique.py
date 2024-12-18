from functools import cache
import os
import json
import random
import argparse
import torch
from tqdm import tqdm
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle

from critiques import *
import glob
# from datasets import concatenate_datasets
import pandas as pd
from peft import AutoPeftModelForCausalLM

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def get_validation_data(args):
    path = f'./data_{args.model_type}/{args.split}/{args.task}.json'
    with open(path) as f:
        final_dataset = json.load(f)
    
    df = pd.DataFrame(final_dataset)

    json_data = json.loads(df.to_json(orient='records'))

    return json_data



def generate_step(prompts, model, tokenizer, temperature, top_p):
    context_tokens = tokenizer(prompts, return_tensors='pt', padding='longest', max_length=4096)
    input_ids = context_tokens.input_ids.to(model.device)
    attention_mask = context_tokens.attention_mask.to(model.device)
    
    if temperature == 0:        
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            use_cache=True,
            max_new_tokens=args.max_len,
            pad_token_id=tokenizer.pad_token_id
        )
    else:
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            use_cache=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_len,
            pad_token_id=tokenizer.pad_token_id
        )
    
    outputs = tokenizer.batch_decode(generation_output[:, input_ids.shape[1]:], skip_special_tokens=False)
    
    results = []
    for output, prompt in zip(outputs, prompts):
        result = output.strip()
        results.append(result)
        
    return results
    
def generate_batch_with_critique(model, tokenizer, prompts, batch_tasks, retry_limit, critique, temperature, top_p):
    batch_size = len(prompts)
    chats = [[p] for p in prompts]
    success = [[] for i in range(batch_size)]
    retry_indices = range(batch_size)
    results = [[] for i in range(batch_size)]
    all_metrics = [[] for i in range(batch_size)]
    
    best_idx = [0 for i in range(batch_size)]
    current_tasks = batch_tasks
    
    for attempt in range(retry_limit+1):
        texts = generate_step(prompts, model, tokenizer, temperature, top_p)
        scores, messages, metrics = critique.batch_critique(current_tasks, texts)

        prompts = []
        new_retry_indices = []
        new_retry_tasks = []
        
        for z, j in enumerate(retry_indices):
            text = texts[z]
            task = batch_tasks[j]
            best = best_idx[j]
            message = messages[z]
            score = scores[z]
            current_metrics = metrics[z]
            if attempt == 0 or score > success[j][best]:
                success[j].append(score)
                results[j].append(text)
                all_metrics[j].append(current_metrics)
                best_idx[j] = attempt
            else:
                if score == 0.2 and score < success[j][best]:
                    g = 5
                success[j].append(success[j][best])
                results[j].append(results[j][best])
                all_metrics[j].append(all_metrics[j][best])
                
            chats[j].append(text)
            if len(message) > 0:
                new_retry_indices.append(j)
                new_retry_tasks.append(task)
                chats[j].append(message)
                #prompts.append("".join(chats[j]))
                # if len(chats[j]) > 20:
                #     prompts.append("".join([chats[j][0]] + chats[j][-20:]))
                # else:
                prompts.append("".join(chats[j]))


        if len(prompts) == 0:
            break         
        
        retry_indices = new_retry_indices
        current_tasks = new_retry_tasks
        
    return results, success, all_metrics
            
def get_critique(task_name, device, generator_tokenizer, model_type):
    if task_name == "num_planning":
        return NumPlanningCritique(device, generator_tokenizer, model_type=model_type, is_training=False) 
    elif  task_name == "common_gen":
        return CommonGenCritique(device, generator_tokenizer, model_type=model_type, is_training=False) 
    elif task_name.startswith("sentiment"):
        return SentimentCritique(device, generator_tokenizer, model_type=model_type, is_training=False) 
    elif task_name == "story_generation":
        return StoryGenerationCritique(device, generator_tokenizer, model_type=model_type, is_training=False) 
    elif task_name == "rationale_generation":
        return RationaleGenerationCritique(device, generator_tokenizer, model_type=model_type, is_training=False) 
    elif task_name == "paraphrase":
        return ParaphraseGenerationCritique(device, generator_tokenizer, model_type=model_type, is_training=False) 
    elif task_name == "style_transfer":
        return StyleTransferCritique(device, generator_tokenizer, model_type=model_type, is_training=False) 
    elif task_name == "clustering":
        return ClusteringCritique(device, generator_tokenizer, model_type=model_type, is_training=False) 
    elif task_name == "panagram":
        return PanagramCritique(device, generator_tokenizer, model_type=model_type, is_training=False) 
    elif task_name == "num_planning_no_feedback":
        return NumPlanningNoFeedbackCritique(device, generator_tokenizer, model_type=model_type, is_training=False)
    elif task_name == "program":
        return ProgramCritique(device, generator_tokenizer, model_type=model_type, is_training=False)
    elif task_name == "mbpp":
        return MbppCritique(device, generator_tokenizer, model_type=model_type, is_training=False)
    elif task_name == "cmg_hard":
        return CmgHardCritique(device, generator_tokenizer, model_type=model_type, is_training=False)
         
def get_result_critique_dict(task, res, success, metrics):
    result_dict = {
        'task':task, 'texts':res, 'success': success, 'metrics': metrics
    }

    return result_dict

def generate(args):    
    initial_instructions = get_validation_data(args)
    if args.model_type == 'llama_2':
        tokenier_model = 'meta-llama/Llama-2-7b-chat-hf'
    elif args.model_type == 'llama_3':
        tokenier_model = 'meta-llama/Meta-Llama-3-8B-Instruct'
        
    tokenizer = AutoTokenizer.from_pretrained(tokenier_model, trust_remote_code=True, max_length=4096, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id

    critique = get_critique(args.task, args.device, tokenizer, args.model_type)
    sep = args.model_prefix_path[-1]
    if sep == "_" or sep == '-':
        model_paths = sorted(glob.glob(f"{args.model_prefix_path}*"), key=lambda s: int(s.split(sep)[-1]), reverse=True)
    else:
        model_paths = [args.model_prefix_path]
    result_dir = f'./{args.model_type}_results/{args.split}/{args.task}/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
                
    for model_path in model_paths:
        display_name = model_path.split("/")[-1]   
        print(f"Initializing model {display_name}")
        
        result_path = result_dir + f'{display_name}_{args.retry_limit}feedback{args.save_suffix}.jsonl'
        
        if os.path.exists(result_path):
            line_num = sum(1 for line in open(result_path))
            start = line_num
            
        else:
            start = 0
            
        if start >= len(initial_instructions):
            continue

        instructions = initial_instructions[start:]
         

        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, cache_dir=cache_dir)
        model.generation_config.temperature=None
        model.generation_config.top_p=None
        critique.generator_tokenizer = tokenizer

        #model.to(args.device)
        model.eval()
        batch_size = args.batch_size
        start = 0
        n = len(instructions)
        all_scores = []
        
        for start in tqdm(range(start, n, batch_size), "Generating"):
            end = min(n, start + batch_size)
            batch_tasks = instructions[start:end]
            batch_prompts = []
            #task = instructions[i]
            for task in batch_tasks:
                batch_prompts.append(task['query'])

            results, success, metrics = generate_batch_with_critique(model, tokenizer, batch_prompts, batch_tasks, args.retry_limit, critique, args.temperature, args.top_p)

            for res, task, scores, current_metrics in zip(results, batch_tasks, success, metrics):
                result_dict = get_result_critique_dict(task, res, scores, current_metrics)
                # save results
                with open(result_path, 'a+') as f:
                    f.write(json.dumps(result_dict))
                    f.write('\n')
                    
                all_scores.append(scores)
            
        attempt_all_avgs = []
        for i in range(args.retry_limit + 1):
            attempt_avgs = []
            for scores in all_scores:
                idx = i if i < len(scores) else -1
                attempt_avgs.append(1 if scores[idx] == 1 else 0)
            print(attempt_avgs)
            attempt_all_avgs.append(sum(attempt_avgs) / len(attempt_avgs))
        
        print(f"Avg = {attempt_all_avgs}")
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--model_prefix_path", default='meta-llama/Llama-2-7b-chat-hf', type=str)
    parser.add_argument("--task", default='common_gen', type=str, choices=["common_gen", "style_transfer", "num_planning", "sentiment", "story_generation", "rationale_generation", "paraphrase", "clustering", "panagram", "program", 'mbpp', 'cmg_hard'])
    parser.add_argument("--split", default='test', type=str, choices=["validation", "test"])
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--save_suffix", default="", type=str)
    parser.add_argument("--model_type", default='llama_3', type=str, choices=["llama_2", "llama_3"])
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--max_len", default=150, type=int)
    parser.add_argument("--temperature", default=0, type=float)    
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument('--retry_limit', default=4, type=int, help='number of feedbacks')

    args = parser.parse_args()
    set_seed(args)

    generate(args)
