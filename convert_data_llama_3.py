import json
import os
from tqdm import tqdm

tasks = [
    "common_gen", 
    "style_transfer", 
    "num_planning", 
    "sentiment", 
    "story_generation", 
    "rationale_generation", 
    "paraphrase",
    "clustering",
    "panagram",
    "program",
    "mbpp",
    "cmg_hard"
]

for split in ['train', 'validation', 'test']:
    for task in tqdm(tasks):
        path_read = f'./data_llama_2/{split}/{task}.json'
        try:
            with open(path_read) as f:
                dataset = json.load(f)
        except:
            continue
        for sample in dataset:
            query = sample['query']
            query = query.replace('[INST] ', '<|start_header_id|>user<|end_header_id|>\n\n')
            query = query.replace(' [/INST]', '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
            sample['query'] = query
        
        save_folder = f'./data_llama_3/{split}'
        path_write = f'{save_folder}/{task}.json'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        
        with open(path_write, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4) 
