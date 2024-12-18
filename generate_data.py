import json
from datasets import Dataset, load_dataset
import pickle
from tqdm import tqdm
import spacy
import os
import random
import csv
import re
import names
from wonderwords import RandomWord
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def read_instruction_template(task):
    with open(f'./instruction_templates/{task}.txt', 'r') as file:
        instruction = file.read()
    return instruction

def generate_common_gen_test_data(instruction_template):
    ds = load_dataset('allenai/commongen_lite', split="train")

    new_ds = []
    for i in tqdm(range(len(ds))):
        sample = ds[i]

        concept_set = sample['concept_set']
        words_list = []
        pos_list = []
        concepts_str_list = []
        for c in concept_set:
            parts = c.split('_')
            pos = 'verb' if parts[1] == 'V' else 'noun'
            word = parts[0]
            words_list.append(word)
            pos_list.append(pos)
            #found_pos_words[word] = pos
            concepts_str_list.append(f'{word}({pos})')
        query = instruction_template.format(concepts=', '.join(concepts_str_list))

        new_sample = {}
        new_sample['query'] = query
        new_sample['words'] = words_list
        new_sample['pos'] = pos_list
        new_ds.append(new_sample)

    return new_ds

def generate_common_gen_train_data(split, instruction_template, limit=None):
    nlp = spacy.load('en_core_web_lg') 
    ds = load_dataset('allenai/common_gen', split=split)
    common_gen_eval = load_dataset('allenai/commongen_lite', split="train")
    used_concepts = set()

    for sample in common_gen_eval:
        concepts = [c.split('_')[0] for c in sample['concept_set']]
        used_concepts.add(frozenset(concepts))

    new_ds = []
    for i in tqdm(range(1, len(ds))):
        sample = ds[i]
        concepts = sample['concepts']
        concept_set = frozenset(concepts)
        if concept_set in used_concepts:
            continue

        new_sample = {}

        used_concepts.add(concept_set)
        doc = nlp(sample['target'])

        # Lemmatization and POS matching
        found_pos_words = {str(tok.lemma_) : str(tok.pos_).lower() for tok in doc if tok.lemma_ in concepts}
        skip  = False
        for v in found_pos_words.values():
            if v != 'noun' and v != 'verb':
                skip = True
                break

        if skip:
            continue
        
        #new_sample['pos'] = found_pos_words
        concepts_str_list = [f'{c[0]}({c[1]})' for c in found_pos_words.items()]
        query = instruction_template.format(concepts=', '.join(concepts_str_list))
        new_sample['id'] = f"common_gen_{split}_{i}"
        new_sample['task'] = 'common_gen'
        new_sample['query'] = query
        new_sample['words'] = list(found_pos_words.keys())
        new_sample['pos'] = list(found_pos_words.values())
        new_sample['ref'] = sample['target']
        new_ds.append(new_sample)

        if limit != None and len(new_ds) == limit:
            break
        
    return new_ds

def generate_common_gen_data(split, instruction_template, limit):
    if split == 'train' or split == 'validation':
        return generate_common_gen_train_data(split, instruction_template, limit)
    
    return generate_common_gen_test_data(instruction_template)



def generate_sentiment_data(task_name, split, instruction_template):   
    if task_name.startswith('sentiment_eval'):
        path = f"./TurnOpt-MTL/raw_data/sentiment/{task_name}.csv"
    else:
        path = f"./raw_data/sentiment/sentiment_{split}.csv"
    with open(path) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        data_read = [row[0] for row in reader]
        dataset = list(data_read[1:])

    samples = []
    for ix, sample in tqdm(enumerate(dataset), f"Generating {split}"):
        sentiment_label = random.randint(1, 5)
        sentiment_label = f"{sentiment_label} stars" if sentiment_label >= 2 else "1 star"
        prompt_text = instruction_template.format(sentiment_label=sentiment_label, product=sample)
        
        sample = {
            "id": f"sentiment_{split}_{ix}",
            "query": prompt_text,
            "label": sentiment_label,
            "task": "sentiment"
        }
        
        samples.append(sample)
    
    return samples

def generate_story_generation_data(split, instruction_template, limit):
    with open(f'./raw_data/story_generation/story_generation_{split}.txt') as f:
        dataset = json.load(f)
        
    samples = []
    for ix, first_sentence in tqdm(enumerate(dataset), f"Generating {split}"):
        if ix == limit:
            break
        prompt_text = instruction_template.format(prefix=first_sentence)

        sample = {
            "id": f"story_generation_{split}_{ix}",
            "query": prompt_text,
            "prefix": first_sentence,
            "task": "story_generation"
        }
        
        samples.append(sample)
        
    return samples


def generate_num_planning_data(split, instruction_template):
    with open(f'./raw_data/num_planning/num_planning_{split}.txt') as f:
        sentences = json.load(f)
    
    samples = []
    for ix, sentence in tqdm(enumerate(sentences), f"Generating {split}"):
        potenital_words = sentence.split(" ")
        words = []
        for word in potenital_words:
            if any(i.isalnum() for i in word):
                words.append(word)
        length = len(words)

        words_to_complete = random.randint(2, min(length - 1, 10))
        prefix_len = length - words_to_complete
        prefix_words = words[:prefix_len]
        last_word = re.sub(r'\W+', '', words[-1])
        
        prefix = ' '. join(prefix_words)
        prompt_text = instruction_template.format(N=words_to_complete, last_word=last_word, prefix=prefix)

        sample = {
            "id": f"num_planning_scores_{split}_{ix}",
            "query": prompt_text,
            "prefix": prefix,
            "last_word": last_word,
            "N": words_to_complete,
            "task": "num_planning_scores",
            "ref": sentence
        }
        
        samples.append(sample)
        
    return samples   


def generate_rationale_generation(split, instruction_template, limit):
    dataset = load_dataset('yangdong/ecqa', split=split)
    
    samples = []
    for ix, item in tqdm(enumerate(dataset), f"Generating {split}"):
        if ix == limit:
            break

        answers = []
        for i in range(1, 6):
            text = item[f'q_op{i}']
            if item['q_ans'] == text:
                correct_answer_index = i
            answers.append(f"{i}. {text}")
        answers_text = '\n'.join(answers)
        correct_answer_text = f'{correct_answer_index}. {item["q_ans"]}'
        question = f"Question: {item['q_text']}\nOptions:\n{answers_text}"
        prompt_text = instruction_template.format(question=question)

        sample = {
            "id": f"rationale_generation_{split}_{ix}",
            "task": "rationale_generation",
            "query": prompt_text,
            "correct_answer": correct_answer_text,
            "student_model_query": f"{question}\n\nAnswer you final answer in the following format - Answer: <answer_index>. <answer_text>\nBackground: ADDED_EXP\nAnswer:",
        }
        
        samples.append(sample)
        
    new_dataset_dict = {key: [] for key in samples[0]}
    for column in new_dataset_dict.keys():
        for sample in samples:
            value = sample[column]
            new_dataset_dict[column].append(value)
    return samples   

def generate_style_transfer_data(split, instruction_template):
    with open(f'./raw_data/style_transfer/style_transfer_{split}.json') as f:
        dataset = json.load(f)
        
    samples = []
    for ix, sample in tqdm(enumerate(dataset), f"Generating {split}"):
        sentence = sample['query']
        prompt_text = instruction_template.format(sentence=sentence)

        sample = {
            "id": f"style_transfer_{split}_{ix}",
            "query": prompt_text,
            "sentence": sentence,
            "ref": sample['ref'],
            "task": "style_transfer"
        }
        
        samples.append(sample)
        
    new_dataset_dict = {key: [] for key in samples[0]}
    for column in new_dataset_dict.keys():
        for sample in samples:
            value = sample[column]
            new_dataset_dict[column].append(value)
    return samples   
    
def generate_paraphrase_data(split, instruction_template):
    with open(f'./raw_data/paraphrase/paraphrase_{split}.json') as f:
        dataset = json.load(f)
     
    samples = []   
    for ix, item in tqdm(enumerate(dataset)):
        source_sentence = item['source']
        exemplar = item['exemplar']
        ref = item['ref']
           
        sample = { 
                    'id': f"paraphrase_{split}_{ix}",
                    'task': 'paraphrase', 
                    'query': instruction_template.format(question=source_sentence, exemplar=exemplar),
                    'source': source_sentence, 
                    'exemplar': exemplar,
                    'ref': ref
                }
        samples.append(sample)
      
    return samples          

def convert_constraints(constraint_dict):
    new_constraints = []
    for student, student_constraints in constraint_dict.items():
        for other_student, match in student_constraints.items():
            new_constraints.append(f'{student}_{other_student}_{str(match)}')
    return new_constraints
    
def generate_clustring_data(split, instruction_template): 
    with open(f'./raw_data/clustering/clustering_{split}.json') as f:
        samples = json.load(f)
        
    for sample in tqdm(samples):
        current_constraints = convert_constraints(sample['constraints'])
        sample['constraints'] = current_constraints
      
    return samples

def generate_panagram_data(split, limit, instruction_template):
    random_words_generator = RandomWord()
    samples = []
    for ix in tqdm(range(limit)):
        word = random_words_generator.word(word_max_length=6, include_parts_of_speech=["nouns"])
        letters_list = list(set(list(word)))
        random.shuffle(letters_list)
        letters_list_str = f'[{", ".join(letters_list)}]'
        prompt_text = instruction_template.format(letters_list=letters_list_str)
        
        sample = {
            "id": f"panagram_{split}_{ix}",
            "query": prompt_text,
            "letters_list": letters_list,
            "ref": word,
            "task": "panagram"
        }
        
        samples.append(sample)
        
    return samples

def generate_mbpp_data(instruction_template):
    data = load_dataset('google-research-datasets/mbpp', 'sanitized', split='test')
    samples = []
    for ix, item in enumerate(data):
        try:
            function_name = item['code'].split(':')[0].split('def ')[1].split('(')[0]
        except:
            print(f"Error with {item['prompt']}")
            continue
        function_signature = item['code'].split(':')[0]
        tests = item['test_list']
        instruction = item['prompt']
        
        query = instruction_template.format(instruction=instruction, function_signature=function_signature)

        
        sample = {
            "id": f"mbpp_{split}_{ix}",
            "query": query,
            "tests": tests,
            "function_name": function_name,
            "function_signature": function_signature,
            "instruction": instruction,
            "task": "mbpp",
            'ref': item['code']
        }
        
        samples.append(sample)
    
    return samples


def generate_cmg_hard(instruction_template):
    raw_data = './raw_data/cmg_hard/cmg_hard.jsonl'
    samples = []
    with open(raw_data, 'r') as file:
        for ix, line in enumerate(file):
            data = json.loads(line)
            concepts = data['concepts']
            concepts_text = ', '.join(concepts) 
            query = instruction_template.format(concepts=concepts_text)
   
            sample = {
                "id": f"cmg_hard_{split}_{ix}",
                "query": query,
                "concepts": concepts,
                "task": "cmg_hard"
            }
            
            samples.append(sample)
    
    return samples

def generate_program_data(instruction_template):
    with open(f'./raw_data/program/program_test.json') as f:
        dataset = json.load(f)
        
    samples = []
    index = 0
    for item in tqdm(dataset):
        if item['category'] != 'numeric':
            continue
        
        mapping = ', '.join([f'f({x}) = {y}' for x, y in zip(item['input_list'], item['output_list'])])
        
        prompt_text = instruction_template.format(mapping=mapping)
        sample = {
            "id": f"program_{split}_{index}",
            "query": prompt_text,
            "input_list": item['input_list'],
            'output_list': item['output_list'],
            'category': item['category'],
            "task": "program"
        }
        
        samples.append(sample)
        index = index + 1
        
    return samples
        
def generate_data(task_name, split, limit):
    if task_name.startswith("sentiment_eval"):
        instruction_template = read_instruction_template('sentiment')
    else:
        instruction_template = read_instruction_template(task_name)
    save_folder = f'./data/{split}'

    if task_name == 'common_gen':
        dataset = generate_common_gen_data(split, instruction_template, limit)
    elif task_name == 'style_transfer':
        dataset = generate_style_transfer_data(split, instruction_template)
    elif task_name.startswith('sentiment'):
        dataset = generate_sentiment_data(task_name, split, instruction_template)
    elif task_name == 'num_planning':
        dataset = generate_num_planning_data(split, instruction_template)
    elif task_name == 'story_generation':
        dataset = generate_story_generation_data(split, instruction_template, limit)
    elif task_name == 'rationale_generation':
        dataset = generate_rationale_generation(split, instruction_template, limit)
    elif task_name == 'paraphrase':
        dataset = generate_paraphrase_data(split, instruction_template)
    elif task_name == 'clustering':
        dataset = generate_clustring_data(split, instruction_template)
    elif task_name == 'panagram':
        dataset = generate_panagram_data(split, limit, instruction_template)
    elif task_name == 'program':
        dataset = generate_program_data(instruction_template)
    elif task_name == 'mbpp':
        dataset = generate_mbpp_data(instruction_template)
    elif task_name == 'cmg_hard':
        dataset = generate_cmg_hard(instruction_template)
    
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        
    with open(f'{save_folder}/{task_name}.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4) 
        
random.seed(10)
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
    ]
splits_with_sizes = {'test': 1000, 'train': 7500, 'validation': 500}
for task in tasks:
    print('*'*40)
    print(f'Generating {task}')
    for split, limit in splits_with_sizes.items():
        print(f'Generating {split}')
        generate_data(task, split, limit)