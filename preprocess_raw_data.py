
from typing import Dict
from tqdm.autonotebook import tqdm
import csv
import torch
import json
import logging
import os
import requests
import zipfile
from tqdm import tqdm
from glob import glob
import pandas as pd
import random
from nltk import tokenize
import json
from nltk.parse import CoreNLPParser
import subprocess
import time
import os
import signal
from subprocess import Popen, PIPE
import socket
from contextlib import closing
from transformers import pipeline
from parser_utils import *
import names
from datasets import load_dataset

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

def unzip(zip_file: str, out_dir: str):
    zip_ = zipfile.ZipFile(zip_file, "r")
    zip_.extractall(path=out_dir)
    zip_.close()
   
def unzip_amazon(): 
    files = glob('./raw_data/sentiment/sentiment_temp/*')
    for file in tqdm(files):
        if not file.endswith('.zip'):
            continue
        dataset = file.split('/')[-1]
        logger.info("Unzipping {} ...".format(dataset))
        out_dir = './raw_data/sentiment/sentiment_temp'
        unzip(file, out_dir)
        

def create_amazon_dataset():
    files = glob('./raw_data/sentiment/sentiment_temp/*')
    li = []
    num_each = 2000

    for file in tqdm(files):
        if file.endswith('.zip'):
            continue
        df = pd.read_csv(file, index_col=None, header=0, sep="\t", error_bad_lines=False, warn_bad_lines=False)
        df = df.sample(n=num_each, random_state=10)
        li.append(df)

    products = pd.concat(li, axis=0, ignore_index=True)
    products = products[['product_title']]
    products = products.sample(frac=1, random_state=10)
    
    train_df = products[:7500]
    validation_df = products[7500:8000]
    test_df = products[8000:9000]
    
    
    for i in range(9):
        eval_variation = products[(9000 + 1000*i):(9000 + 1000*(i+1))]
        eval_variation.to_csv(f'./raw_data/sentiment/sentiment_eval_{i+1}.csv', sep=',', index=False, encoding='utf-8')

    
    # train_df.to_csv('./raw_data/sentiment/sentiment_train.csv', sep=',', index=False, encoding='utf-8')
    # validation_df.to_csv('./raw_data/sentiment/sentiment_validation.csv', sep=',', index=False, encoding='utf-8')
    # test_df.to_csv('./raw_data/sentiment/sentiment_test.csv', sep=',', index=False, encoding='utf-8')


    
def get_num_planning_sentences(stories, limit):
    all_sentences = []
    for story in stories:
        sentences = tokenize.sent_tokenize(story)[1:]
        for sentence in sentences:
            potenital_words = sentence.split(" ")
            words = []
            for word in potenital_words:
                if any(i.isalnum() for i in word):
                    words.append(word)
            length = len(words)
            if length <= 2:
                continue
            all_sentences.append(sentence)
    
    random.shuffle(all_sentences)
    return all_sentences[:limit]
    
def create_num_planning_dataset():
    with open("./data/reddit_short_stories.txt", encoding='utf-8') as file:
        data_read = file.read()
    
    stories = data_read.split('<eos>\n<sos> ')
    stories = [s.replace('<nl> ', '').replace('<sos> ', '') for s in stories]
    
    random.shuffle(stories)
    train_stories = stories[:2800]
    validation_stories = stories[2800:3300]
    test_stories = stories[3300:]
    
    train = get_num_planning_sentences(train_stories, 7500)
    validation = get_num_planning_sentences(validation_stories, 500)
    test = get_num_planning_sentences(test_stories, 1000)
    
    
    with open('./raw_data/num_planning/num_planning_train.txt', 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False, indent=4)
        
    with open('./raw_data/num_planning/num_planning_validation.txt', 'w', encoding='utf-8') as f:
        json.dump(validation, f, ensure_ascii=False, indent=4)
        
    with open('./raw_data/num_planning/num_planning_test.txt', 'w', encoding='utf-8') as f:
        json.dump(test, f, ensure_ascii=False, indent=4)
        
def create_story_generation():    
    with open("./data/ROCStories_winter2017.csv") as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')

        dataset = [row[2] for row in reader]
        dataset = list(set(dataset[1:]))
    random.shuffle(dataset)
    train = dataset[:7500]
    validation = dataset[7500:8000]
    test = dataset[8000:9000]
    
    with open('./raw_data/story_generation/story_generation_train.txt', 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False, indent=4)
        
    with open('./raw_data/story_generation/story_generation_validation.txt', 'w', encoding='utf-8') as f:
        json.dump(validation, f, ensure_ascii=False, indent=4)
        
    with open('./raw_data/story_generation/story_generation_test.txt', 'w', encoding='utf-8') as f:
        json.dump(test, f, ensure_ascii=False, indent=4)

def create_style_transfer_data():
    dataset = []

    for i, line in enumerate(tqdm(open('./data/biased.full'))):
        parts = line.strip().split("\t")

        if len(parts) == 7:
            rev_id, pre_tok, post_tok, pre_raw, post_raw, pos, rels = parts

        elif len(parts) == 5:
            rev_id, pre_tok, post_tok, pre_raw, post_raw = parts

        else:
            print(f"Skipped entry: {i}")

        
        dataset.append({'query': pre_raw, 'ref': post_raw})
        
    random.shuffle(dataset)
    random.shuffle(dataset)
    train = dataset[:7500]
    validation = dataset[7500:8000]
    test = dataset[8000:9000]
    
    with open('./raw_data/style_transfer/style_transfer_train.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False, indent=4)
        
    with open('./raw_data/style_transfer/style_transfer_validation.json', 'w', encoding='utf-8') as f:
        json.dump(validation, f, ensure_ascii=False, indent=4)
        
    with open('./raw_data/style_transfer/style_transfer_test.json', 'w', encoding='utf-8') as f:
        json.dump(test, f, ensure_ascii=False, indent=4)
    

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
        
def run_server():
        port = find_free_port()
        command1 = subprocess.Popen([f'./apps/stanford-corenlp-full-2018-10-05 ; java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port {port}'], shell=True, stdout=subprocess.PIPE)
        time.sleep(10)
        return port
    
def reinit_server(port):
        process = Popen(["lsof", "-i", ":{0}".format(port)], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        for process in str(stdout.decode("utf-8")).split("\n")[1:]:       
            data = [x for x in process.split(" ") if x != '']
            if (len(data) <= 1):
                continue

            os.kill(int(data[1]), signal.SIGKILL)
        
        time.sleep(5)
        port = run_server()
        parser = CoreNLPParser(url=f'http://localhost:{port}')
        return port, parser
        
def preprocess_paraphrase():
    task = 'qqppos'
    port = run_server()
    parser = CoreNLPParser(url=f'http://localhost:{port}')

    for split in ["train", "test", "val"]:
        source_file = f'./raw_data/{task}/{split}.source'
        tgt_file = f'./raw_data/{task}/{split}.target'
        samples = []
        i = 0
        with open(source_file) as source_file:
            with open(tgt_file) as tgt_file:
                for source_line, target_line in tqdm(zip(source_file, tgt_file)):
                    if split == "train" and i == 7500 or split == "val" and i == 500:
                        break
                    source_line = source_line.rstrip()
                    source_sentence = source_line.split('<sep>')[0]
                    target_sentece = target_line.split('<sep>')[1]
                    source_sentence = source_sentence.strip()
                    target_sentece = target_sentece.strip()
                    
                    try:
                        trees = list(parser.raw_parse_sents([target_sentece]))
                    except:
                        port, parser = reinit_server(port)
                        trees = list(parser.raw_parse_sents([target_sentece]))
                    
                    
                    assert len(trees) == 1

                    for tree in list(trees):
                        tree = list(tree)[0]
                        parse = convert_str(str(tree))

                        pure_parse = ''
                        for p in parse.split('\n'):
                            new_p = convert_str(p)
                            pure_parse += ' ' + new_p.strip()
                
                        exemplar = trim_str(pure_parse, 3)
                        sample = {
                                    'source': source_sentence, 
                                    'exemplar': exemplar,
                                    'ref': target_sentece
                                }
                        samples.append(sample)
                        
                    i = i + 1
        
        if split == "val":
            split = "validation"
        with open(f'./raw_data/paraphrase/paraphrase_{split}.json', 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=4)      


def parse_text(text, parser, port):
    try:
        trees = list(parser.raw_parse_sents([text]))
    except:
        port, parser = reinit_server(port)
        trees = list(parser.raw_parse_sents([text]))
            
    for tree in list(trees):
            tree = list(tree)[0]
            parse = str(tree)#convert_str(str(tree))

            pure_parse = ''
            for p in parse.split('\n'):
                new_p = p#convert_str(p)
                pure_parse += ' ' + new_p.strip()
    
            parse = pure_parse
            #parse = trim_str(pure_parse, 3)   
            
    # Only one tree for this function
    return parse, port, parser      
            
def preprocess_paraphrase_evaluation():
    port = run_server()
    parser = CoreNLPParser(url=f'http://localhost:{port}')
    model = 'paraphrase_0_210_9feedback'
    results_path = f'./test_results/paraphrase/{model}.jsonl'
        
    with open(results_path) as f:
        results = [json.loads(line) for line in f]
        
    results_for_eval = []
        
    for result in tqdm(results):
        data = result['task']
        generated_text = result['texts'][-1].strip()
        if '?' in generated_text:
            index = generated_text.index('?')
            generated_text = generated_text[:(index+1)]
        elif '.' in generated_text:
            index = generated_text.index('.')
            generated_text = generated_text[:(index+1)]
        generated_text_parse, port, parser = parse_text(generated_text, parser, port)
        pure_parse = data['exemplar']
        ref = data['ref']
        ref_parse, port, parser = parse_text(ref, parser, port)
        results_for_eval.append([generated_text, ref, generated_text_parse, ref_parse, pure_parse])
        #input_line, ref_line, input_parse, ref_parse,temp_parse

    with open(f'./raw_data/paraphrase/results_{model}.json', 'w', encoding='utf-8') as f:
        json.dump(results_for_eval, f, ensure_ascii=False, indent=4)             

def constraints_to_string(constraints):
    text_list = []
    counter = 1
    for student, constraint_dict in constraints.items():
        for other_student, match in constraint_dict.items():
            if match:
                text_list.append(f'{counter}. {student} wants to be grouped with {other_student}.')
            else:
                text_list.append(f'{counter}. {student} doesn\'t want to be grouped with {other_student}.')
            counter += 1
    return '\n'.join(text_list)

def generate_clustring_data(split, limit): 
    with open(f'./instruction_templates/clustering.txt', 'r') as file:
        instruction_template = file.read()
    samples = []   
    for ix in tqdm(range(limit)):
        student_number = random.randint(4, 10)
        students = [names.get_first_name() for j in range(student_number)]
        members_count = 2
        groups = []
        for i in range(len(students) // members_count + 1):
            group = students[i*members_count:i*members_count + members_count]
            groups.append(group)
            
        if len(groups[-1]) == 1 or len(groups[-1]) == 0:
            groups[-2].extend(groups[-1])
            del groups[-1]
        
        random.shuffle(students)
        constraints_number = random.randint(1, len(groups)-1)
        constraint_dict = {}
        for j in range(constraints_number):
            group = groups[j]
            match = random.choice([True, False])
            student = group[0]
            if match:
                other_student = group[1]
            else:
                other_group_idx = random.randint(j+1, len(groups)-1)
                other_group = groups[other_group_idx]
                other_student = random.choice(other_group)
            if student not in constraint_dict:
                constraint_dict[student] = {}
            constraint_dict[student][other_student] = match
        
        constraints_text = constraints_to_string(constraint_dict)
        students_text = ', '.join(students)
   
        sample = { 
                    'id': f"clustering_{split}_{ix}",
                    'task': 'clustering', 
                    'query': instruction_template.format(students=students_text, constraints=constraints_text),
                    'students': students, 
                    'constraints': constraint_dict
                }
        samples.append(sample)
        
    with open(f'./raw_data/clustering/clustering_{split}.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)     



# The following code is taken from Big Bench - Program Synthesis task:
# https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/program_synthesis
"""Python program generation task. Given a list of input/outputs, find the
simplest python function that can satisfy the input output relationship."""

import numpy as np
import random
import string

from collections import defaultdict
from RestrictedPython import compile_restricted
from RestrictedPython import safe_builtins
from typing import List, Any, Dict

COMPILE_SCORE = 20
CORRECTNESS_SCORE_MAX = 60
LENGTH_MAX_SCORE = 25
NUM_TRIALS = 4  # The number of times same program is used with different inputs


def f(x):
    """Dummy function. The model should define this function to satisfy given input/outputs."""
    return x + 42


def bizfuzz(n: int) -> str:
    if n % 5 == 0:
        return "biz"
    if n % 4 == 0:
        return "fuzz"
    return str(n)


def add_noise(input_list) -> List[Any]:
    """This adds extra random inputs to the input to prevent memorization, and returns it shuffled.

    The final inputs will contain manually written indicative inputs + randomly generated input. The indicative
    inputs will ensure the program is easier to determine."""
    result = input_list
    if isinstance(input_list, list):
        if isinstance(input_list[0], int) or isinstance(input_list[0], float):
            result.extend(random.sample(range(20), 4))
        elif isinstance(input_list[0], str):
            result.append(''.join(random.choice(string.ascii_letters) for _ in range(5)))
        elif isinstance(input_list[0], list) and len(input_list[0]) > 0 and isinstance(input_list[0][0], bool):
            result.append([random.choice([True, False]) for _ in range(3)])
    random.shuffle(result)
    return result


def create_query(input_list: List[Any], output_list: List[Any]) -> str:
    return ', '.join([f'{x} to {y}' for x, y in zip(input_list, output_list)])


STR_ARRAY = ['abc', 'Abc', 'ABC', '123', 'abc123', 'ab#', 'efg', 'Efg', 'EFG', '789', 'e*g']
BOOL_ARRAY = [[True, True], [True, False], [False, True], [False, False]]
MIXED_TYPE_ARRAY = [5, 'hat', 2.2, [], {}, 82, 'cat', True, False, [1, 2], {'a'}]



# This dataset contains a set of indicative inputs and the function to be synthesized organized by category.
# Each category is a subtask
PROGRAM_DATASET = {
    'numeric': [
        [[0, 1, 2], lambda x: x],
        [[6, 7, 3], lambda x: x + 1],
        [[6, 7, 3], lambda x: x * 12],
        [[6.0, 7.0, 3.0], lambda x: x / 3],
        [[6.0, 7.0, 3.0], lambda x: 88.0 / (x + 1)],
        [[6.6, 7.8, 3.2], lambda x: round(x)],
        [[6, 7, 8, 9, 10, 11, 12], lambda x: x // 3],
        [[2, 3, 4], lambda x: x ** 3],
        [[2, 3, 4, 6], lambda x: x ** 2 - 6 * x],
        [[2, 3, 4], lambda x: x * x],
        [[0, 1, 2], lambda x: 4 * x * x + 3 * x - 62],
        [[0, 1, 2], lambda x: 5],
        [[-2, -1, 0, 1, 2], lambda x: abs(x)],
        [list(range(8)), lambda x: bin(x)],
        [[9, 10, 11, 11.1, 12, 13, 14, 15], lambda x: x > 11],
        [[0, 1, 2, 3, 4, 88, 89], lambda x: 234817 if x % 2 == 0 else 918276],
        [[-1000, -100, 0, 32, 33, 55, 999], lambda x: 8 if x > 32 else 7],
        [[2], lambda x: list(range(x))],
    ],
    'string': [
        [STR_ARRAY, lambda x: x.upper()],
        [['o', 'r', 'd', 'A'], lambda x: ord(x[0])],
        [STR_ARRAY, lambda x: x.replace('a', 'b')],
        [STR_ARRAY, lambda x: x.isalnum()],
        [STR_ARRAY, lambda x: x.isalpha()],
        [STR_ARRAY, lambda x: x.islower()],
        [STR_ARRAY, lambda x: x.isupper()],
        [STR_ARRAY, lambda x: x.isnumeric()],
        [STR_ARRAY, lambda x: x.istitle()],
        [STR_ARRAY, lambda x: x.split('b')],
        [STR_ARRAY, lambda x: x == 'EFG'],
        [STR_ARRAY, lambda x: x[2] == 'c'],
        [STR_ARRAY, lambda x: x.lower() == 'abc'],
        [['a', 'cde', 'hijklm', 'n', 'zzzzzzzz'], lambda x: len(x)],
        [list(zip(STR_ARRAY, STR_ARRAY[::-1])), lambda x: x[0] + x[1]],
    ],
    'collection': [
        [[[5, 1, 8], [-2, -22, -3], [-4, 6, -2]], lambda x: min(x)],
        [[[5, 1, 8], [-2, -22, -3], [99, 9, 0]], lambda x: sorted(x)],
        [[[5, 1, 8], [-2, -22, -3], [-4, 6, -2]], lambda x: max(x)],
        [[[4], [4, 2], [100, 1], [5, 1, 8], [-2, -22, -3], [-4, 6, -2]], lambda x: sum(x)],
        [[['abc', 22], [5, 1, 8], [-2, -22, -3, 1], [-2, -22, -3], [-4, 6, -2]], lambda x: 1 in x],
        [[['abc', 22], [5, 1, 8], [-2, -22, -3, 1], [], [88], [-2, -22, -3], [-4, 6, -2]], lambda x: len(x)],
        [[[5, 1, 8], [-2, -22, -3], [-4, 6, -2]], lambda x: max(x) + 100],
        [[[5, 1, 8], [-2, -22, -3], [-4, 6, -2]], lambda x: x[2]],
        [[[5, 1, 8], [-2, -22, -3], [-4, 6, -2]], lambda x: x[2] * 2],
        [list(range(20)), bizfuzz],
    ],
    'logical': [
        [[5, 8, 'hi', -3, 6], lambda x: True],
        [[True, False], lambda x: not x],
        [[[True, True, False], [True, True, True], [False, False, False]], lambda x: any(x)],
        [BOOL_ARRAY, lambda x: x[0]],
        [BOOL_ARRAY, lambda x: x[1]],
        [BOOL_ARRAY, lambda x: not x[0]],
        [BOOL_ARRAY, lambda x: not x[1]],
        [BOOL_ARRAY, lambda x: x[0] and x[1]],
        [BOOL_ARRAY, lambda x: x[0] or x[1]],
        [BOOL_ARRAY, lambda x: (not x[0]) and x[1]],
        [BOOL_ARRAY, lambda x: (not x[1]) and x[0]],
        [BOOL_ARRAY, lambda x: x[0] ^ x[1]],
        [BOOL_ARRAY, lambda x: not (x[0] and x[1])],
        [BOOL_ARRAY, lambda x: not (x[0] or x[1])],
        [BOOL_ARRAY, lambda x: all(x)],
    ],
    'type': [
        [MIXED_TYPE_ARRAY, lambda x: type(x)],
        [MIXED_TYPE_ARRAY, lambda x: isinstance(x, int)],
        [MIXED_TYPE_ARRAY, lambda x: isinstance(x, str)],
        [MIXED_TYPE_ARRAY, lambda x: isinstance(x, float)],
        [MIXED_TYPE_ARRAY, lambda x: isinstance(x, list)],
        [MIXED_TYPE_ARRAY, lambda x: isinstance(x, set)],
    ]
}


def create_program_synthesis_data():
    NUM_TRIALS = 4
    score_data = []
    programs_by_category = defaultdict(list)
    random.seed(10)
    programs_by_category = PROGRAM_DATASET
    
    samples = []

    for category, category_programs in programs_by_category.items():
        for program_index, (original_input_list, actual_function) in enumerate(category_programs):
            for trial in range(NUM_TRIALS):
                input_list = add_noise(original_input_list)
                output_list = [actual_function(x) for x in input_list]
                
                input_list = [str(x) for x in input_list]
                output_list = [str(x) for x in output_list]

                sample = {'input_list': input_list, 'output_list': output_list, 'category': category }
                samples.append(sample)
                
    
    with open(f'./raw_data/program/program_test.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)   
                
    return score_data
         
      
