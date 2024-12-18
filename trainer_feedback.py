from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, set_seed
from chat_ppo_trainer import PPOTrainer
import datetime
import json
from datasets import Dataset, concatenate_datasets
from critiques import *

tqdm.pandas()

is_debug = False

def get_rewards(critique, batch, responses):
    rewards_list, messages, _ = critique.batch_critique(batch, responses)
    return rewards_list, messages

def left_pad(t, pad_value):
    return torch.nn.utils.rnn.pad_sequence([x.flip(dims=[1]).squeeze(0) for x in t], batch_first=True, padding_value=pad_value).flip(dims=[1]) 

def prepare_step_inputs(chats_tensors, tokenizer, device):
    all_chats_input_ids_list = []
    all_chats_mask_list = []
    for chat_tensors in chats_tensors:
        chat_mask_list = []
        for i, chat_tensor in enumerate(chat_tensors):
            if i % 2 == 0:
                message_mask = torch.zeros_like(chat_tensor)
                if len(message_mask[0]) > 0:
                    message_mask[0][-1] = 1
            else:
                message_mask = torch.ones_like(chat_tensor)
                if len(message_mask[0]) > 0:
                    message_mask[0][-1] = 0
            chat_mask_list.append(message_mask)

        chat_input_ids = torch.cat([t.to(device) for t in chat_tensors], dim=1)    
        chat_mask = torch.cat([t.to(device) for t in chat_mask_list], dim=1)    
        all_chats_input_ids_list.append(chat_input_ids)
        all_chats_mask_list.append(chat_mask)


    chats_input_ids = left_pad(all_chats_input_ids_list, tokenizer.pad_token_id).to(device)
    mask = left_pad(all_chats_mask_list, 0).to(device)

    return chats_input_ids, mask[:, :-1]



def build_dataset(tokenizer, ds):
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["query"], max_length=1024, padding='longest')
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])        


def convert_dataset_from_json(dataset):
    new_dataset_dict = {key: [] for key in dataset[0]}
    for column in new_dataset_dict.keys():
        for sample in dataset:
            value = sample[column]
            new_dataset_dict[column].append(value)
    return Dataset.from_dict(new_dataset_dict)        

def load_datasets(task, model_type):
    if task == 'meta_learning':
        tasks = ['num_planning', 'paraphrase', 'rationale_generation', 'sentiment', 'story_generation']
        datasets = []
        for task in tasks:
            path = f'ץ/data_{model_type}/train/{task}.json'
            with open(path) as f:
                dataset = json.load(f)
            dataset = convert_dataset_from_json(dataset)
            datasets.append(dataset)
        mtl_dataset = concatenate_datasets(datasets)
        return mtl_dataset
    else:
        path = f'./data_{model_type}/train/{task}.json'
        with open(path) as f:
            dataset = json.load(f)
        dataset = convert_dataset_from_json(dataset)
        return dataset

def get_critique(task_name, device, generator_tokenizer, model_type):
    if task_name == "meta_learning":
        return MetaLearningCritique(device, generator_tokenizer, model_type=model_type, is_training=True) 
    elif task_name == "clustering":
        return ClusteringCritique(device, generator_tokenizer, model_type=model_type, is_training=True) 
    elif task_name == "num_planning":
        return NumPlanningCritique(device, generator_tokenizer, model_type=model_type, is_training=True) 
    elif task_name == "common_gen":
        return CommonGenCritique(device, generator_tokenizer, model_type=model_type, is_training=True) 
    elif task_name == "sentiment":
        return SentimentCritique(device, generator_tokenizer, model_type=model_type, is_training=True) 
    elif task_name == "story_generation":
        return StoryGenerationCritique(device, generator_tokenizer, model_type=model_type, is_training=True) 
    elif task_name == "rationale_generation":
        return RationaleGenerationCritique(device, generator_tokenizer, model_type=model_type, is_training=True) 
    elif task_name == "paraphrase":
        return ParaphraseGenerationCritique(device, generator_tokenizer, model_type=model_type, is_training=True) 
    elif task_name == "style_transfer":
        return StyleTransferCritique(device, generator_tokenizer, model_type=model_type, is_training=True)    
    elif task_name == "panagram":
        return PanagramCritique(device, generator_tokenizer, model_type=model_type, is_training=True)    
    elif task_name == "num_planning_no_feedback":
        return NumPlanningScoresCritique(device, generator_tokenizer, model_type=model_type, is_training=True) 
    elif task_name == "program":
        return ProgramCritique(device, generator_tokenizer, model_type=model_type, is_training=True)
    elif task_name == "mbpp":
        return MbppCritique(device, generator_tokenizer, model_type=model_type, is_training=True)
    elif task_name == "cmg_hard":
        return CmgHardCritique(device, generator_tokenizer, model_type=model_type, is_training=True)

@dataclass
class ScriptArguments:
    # LoraConfig
    task: Optional[str] = field(default='meta_learning', metadata={"help": "The task to train"})
    device: Optional[str] = field(default='cuda', metadata={"help": "cuda or cpu"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "whether to use peft"})
    max_new_tokens: Optional[int] = field(default=150, metadata={"help": "the lora r parameter"})
    retry_limit: Optional[int] = field(default=3, metadata={"help": "the lora r parameter"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})
    starting_step: Optional[int] = field(default=0, metadata={"help": "For loaded checkpoints"})
    model_type: Optional[str] = field(default='llama_3', metadata={"help": "Can be llama_2 or llama_3"})

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig))
    args, ppo_config = parser.parse_args_into_dataclasses()
    ppo_config.gradient_accumulation_steps = 1
    ppo_config.log_with = 'wandb'
    ppo_config.tracker_project_name = f'{args.task}_rl'
    ppo_config.remove_unused_columns = False
    ppo_config.tracker_kwargs = {'name': 'baseline_{date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now() )}
    dataset = load_datasets(args.task, args.model_type)

    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    critique = get_critique(args.task, args.device, tokenizer, args.model_type)
    
    set_seed(ppo_config.seed)

    peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
        )
    ref_model = None
    device_map = {"": Accelerator().local_process_index}
    
    dataset = build_dataset(tokenizer, dataset)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
            ppo_config.model_name,
            device_map=device_map,
            peft_config=peft_config,
            cache_dir=cache_dir,
            is_trainable=True
        )
    
    ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": args.max_new_tokens,
    }

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

    counter = 0
    retry_limit = args.retry_limit
    print(f"Retry limit = {retry_limit}")

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        counter = counter+1
        
        if counter < args.starting_step:
            continue
        
        query_tensors = batch["input_ids"]

        # Get response
        response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, **generation_kwargs
        )
        batch_samples = [{key: batch[key][i] for key in batch if key != 'input_ids'} for i in range(len(batch['query']))]

        responses = tokenizer.batch_decode(response_tensors)
        responses = critique.extract_texts(batch_samples, responses)

        queries = []
        batch_messages = []
        rewards = []
        chat_tensors = [[q.unsqueeze(0)] for q in query_tensors] 
        response_tensors = []

        for i in range(len(batch['query'])):
            query = batch['query'][i]
            response = responses[i]
            response_tensor = tokenizer.encode(response, max_length=1024, padding='longest', return_tensors='pt', add_special_tokens=False).to(device)
            response_tensors.append(response_tensor)
            chat_tensors[i].append(response_tensor)
            batch_messages.append([query, response])
            queries.append(query)
            

        rewards_list, messages = get_rewards(critique, batch_samples, responses)

        retry_indices = range(len(batch['query']))

        for retry_number in range(retry_limit):
            # preparing the next retry
            new_queries = []
            new_retry_indices = []
            current_samples = []

            for z, j in enumerate(retry_indices):
                response = responses[j]
                new_queries_texts = []
                current_messages = batch_messages[j]

                if len(messages[z]) > 0:
                    current_samples.append(batch_samples[j])
                    revise_message = messages[z]
                    current_messages.append(revise_message)
                    revise_message_tensor = tokenizer.encode(revise_message, max_length=1024, padding='longest', return_tensors='pt', add_special_tokens=False).to(device)
                    chat_tensors[j].append(revise_message_tensor)
                    prompt = ''.join(current_messages)
                    new_queries.append(prompt)

                    new_retry_indices.append(j)


            if len(new_retry_indices) == 0:
                break
            else:
                new_query_tensors = list(left_pad([torch.cat([t.to(device) for t in chat_tensors[k]], dim=1) for k in new_retry_indices], tokenizer.pad_token_id))
                new_response_tensors = ppo_trainer.generate(
                    new_query_tensors, return_prompt=False, **generation_kwargs
                )
                new_responses = tokenizer.batch_decode(new_response_tensors)
                new_responses = critique.extract_texts(current_samples, new_responses)

                new_rewards_list, messages = get_rewards(critique, current_samples, new_responses)
                retry_indices = new_retry_indices

                for z, j in enumerate(retry_indices):
                    rewards_list[j] = max(new_rewards_list[z], rewards_list[j])
                    response = new_responses[z]

                    response_tensor = tokenizer.encode(response, max_length=1024, padding='longest', return_tensors='pt', add_special_tokens=False).to(device)
                    response_tensors[j] = response_tensor
                    query_tensors[j] = new_query_tensors[z]
                    responses[j] = new_responses[z]
                    queries[j] = new_queries[z]

                    current_messages = batch_messages[j]
                    current_messages.append(response)
                    chat_tensors[j].append(response_tensor)
                

        rewards = [torch.tensor(reward, device=device, dtype=torch.float) for reward in rewards_list]
        chats_input_ids, masks = prepare_step_inputs(chat_tensors, tokenizer, device)
        
        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, chats_input_ids, masks, rewards)
        batch['response'] = responses
        batch['query'] = queries
        ppo_trainer.log_stats(stats, batch, rewards)

        if counter%10 == 0 and not is_debug and Accelerator().local_process_index == 0:
            ppo_trainer.save_pretrained(f"./runs/{args.model_type}_{args.task}_{retry_limit}_{counter}", safe_serialization=True, max_shard_size='4GB')

    if not is_debug:
        ppo_trainer.save_pretrained(f"./runs/{args.model_type}_{args.task}_{retry_limit}", safe_serialization=True, max_shard_size='4GB')