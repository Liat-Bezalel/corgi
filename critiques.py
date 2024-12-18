from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoModel, T5ForConditionalGeneration
import spacy
import torch
import re
from metrics import rep_n, coherence
from nltk.parse import CoreNLPParser
import subprocess
import time
import os
import signal
from subprocess import Popen, PIPE
import socket
from contextlib import closing
from transformers import pipeline
from ted_e_metric import ted_e_3, get_pure_parse
import re
import logging
import enchant
from RestrictedPython import compile_restricted
from RestrictedPython import safe_builtins
import os
import time
import threading
import sys
import queue
import multiprocessing
import time

os.environ['TOKENIZERS_PARALLELISM'] = "false"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

class CritiqueBase(object):
    def __init__(self, device, generator_tokenizer, model_type="llama_2", is_training=False) -> None:
        print("Initialize %s"%self.__class__.__name__)
        self.device = device
        self.model_type = model_type
        self.is_training = is_training
        self.generator_tokenizer = generator_tokenizer
    
    def batch_critique(self, tasks, texts):
        if not self.is_training:
            texts = self.extract_texts(tasks, texts)
            
        return self.batch_critique_impl(tasks, texts)
    
    def extract_texts(self, tasks, texts):
        return [self.extract_text(texts[i], tasks[i]) for i in range(len(texts))]
    
    
    def batch_critique_impl(self, tasks, texts):
        raise NotImplementedError("Please Implement this method")
    
    def extract_text(self, text, task):
        raise NotImplementedError("Please Implement this method")
    
    def combine_messages(self, messages, suffix):
        current_message = ""
        counter = 0
        for message in messages:
            if len(message) == 0:
                continue
            if len(current_message) == 0:
                current_message = message
            else:
                message = message[0].lower() + message[1:]
                counter += 1
                if counter == 1:
                    addition_word = "In addition"
                else:
                    addition_word = "Moreover"
                current_message = f"{current_message} {addition_word}, {message}"
       
        if len(current_message) > 0:
            if self.model_type == "llama_2":
                return f"[INST]\n- Feedback: {current_message} Please Try again. [/INST]\n- Output:{suffix}"      
            elif self.model_type == "llama_3":
                return f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\n- Feedback: {current_message} Please Try again.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\n- Output:{suffix}"      
 
        return current_message
        

class CmgHardCritique(CritiqueBase):
    def __init__(self, device, generator_tokenizer, model_type="llama_2", is_training=False) -> None:
        CritiqueBase.__init__(self, device, generator_tokenizer, model_type, is_training)
        self.nlp = spacy.load('en_core_web_lg')

    def extract_text(self, text, task):
        if '.' in text:
            index = text.index('.')
            text = text[:(index+1)]
        return text
    
    def analyze_words(self, concepts, sentence):
        doc = self.nlp(sentence)
        
        # Lemmatization and POS matching
        found_words = [tok.lemma_ for tok in doc if tok.lemma_ in concepts]

        # print("Number of words used from the given list: ", len(found_words))
        # print("Number of words with correct POS: ", len(found_pos_words))
        
        # take unique items 
        found_words = list(set(found_words))
        return found_words
    
    def get_revise_message(self, concept_set, found_words):
        cover = len(concept_set) == len(found_words)
        missing_words = []
        words = concept_set
        messages = []
        if not cover:
            for word in words:
                if word not in found_words:
                    missing_words.append(word)
            messages.append(f"You are missing the following concepts: {', '.join(missing_words)}.")
            
        return self.combine_messages(messages, "")
        
    def batch_critique_impl(self, tasks, texts):
        scores = []
        messages = []
        metrics = []
        
        for task, text in zip(tasks, texts):
            concepts = task['concepts']
            found_words = self.analyze_words(concepts, text)
            found_words_score = len(found_words) / len(concepts)
            score = 1 if len(found_words) == len(concepts) else 0
            scores.append(score)
            message = self.get_revise_message(concepts, found_words)
            messages.append(message)
            metrics.append({'cover': found_words_score})
            
        return scores, messages, metrics

        
class CommonGenCritique(CritiqueBase):
    def __init__(self, device, generator_tokenizer, model_type='llama_2', is_training=False) -> None:
        CritiqueBase.__init__(self, device, generator_tokenizer, model_type, is_training)
        model_name = "google/gemma-1.1-7b-it"
        self.critique_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="balanced_low_0",
            load_in_4bit=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.nlp = spacy.load('en_core_web_lg')
        self.common_sense_template = 'user\nDoes the following sentence describe a common sense scenario?\n\n"{}"\n\nAnswer: no / yes. Write an explanation with the format of Explanation: <Explanation>. Your explanation should be one sentence.\nmodel\nAnswer:'


    def extract_text(self, text, task):
        if '.' in text:
            index = text.index('.')
            text = text[:(index+1)]
        return text
    
    def analyze_words(self, input_dict, sentence):
        doc = self.nlp(sentence)
        
        # Lemmatization and POS matching
        found_words = [tok.lemma_ for tok in doc if tok.lemma_ in input_dict.keys()]
        found_pos_words = [tok.lemma_+"_"+tok.pos_ for tok in doc if tok.lemma_ in input_dict.keys() and input_dict[tok.lemma_].lower() in tok.pos_.lower()]
        
        # take unique items 
        found_words = list(set(found_words))
        found_pos_words = list(set(found_pos_words))
        return found_words, found_pos_words
    
    def common_sense_score(self, responses):
        prompts = [self.common_sense_template.format(response) for response in responses]
        outputs = self.hf_generate(self.critique_model, self.tokenizer, prompts, 40)
        scores = []
        feedback_messages = []
        for input, output in zip(prompts, outputs):
            idx = output.rfind('Answer:') + len('Answer:')
            answer = output[idx:].strip()
            is_common_sense = 1 if answer.lower().startswith('yes') else 0
            feedback = ""
            if is_common_sense == 0:
                match = re.search(r'.*Explanation: (?P<explanation>.*)', answer)
                if match:
                    feedback = f"Your sentence doesn't describe a common sense scenario: {match.groupdict()['explanation']}"
                else:
                    if answer.startswith("No."):
                        explanation = answer[3:].strip()
                        feedback = f"Your sentence doesn't describe a common sense scenario: {explanation}"
                    else:
                        logger.info(f"Error with output: {answer}")
            feedback_messages.append(feedback)
            scores.append(1 if answer.lower().startswith('yes') else 0)
        return scores, feedback_messages
    
    def get_revise_message(self, concept_set, found_words, found_pos_words, is_natural, cs_feedback):
        cover = len(concept_set) == len(found_words)
        pos_words = len(concept_set) == len(found_pos_words)
        missing_words = []
        wrong_pos = []
        words = concept_set.keys()
        messages = []
        if not cover:
            for word in words:
                if word not in found_words:
                    missing_words.append(word)
            messages.append(f"You didn't use the following concepts: {', '.join(missing_words)}.")
        if not pos_words:
            for word, word_pos in concept_set.items():
                if f'{word}_{word_pos.upper()}' not in found_pos_words:
                    wrong_pos.append(f'{word} as a {word_pos.lower()}')

            messages.append(f"You should correct the part of speech: {', '.join(wrong_pos)}.")

        if not is_natural:
            messages.append(cs_feedback)
            
        return self.combine_messages(messages, "")
        
    def batch_critique_impl(self, tasks, texts):
        scores = []
        messages = []
        metrics = []
        natrual_scores, feedback_messages = self.common_sense_score(texts)
        
        for task, text, natrual_score, cs_feedback in zip(tasks, texts, natrual_scores, feedback_messages):
            words = task['words']
            pos = task['pos']
            concept_set = {words[k]: pos[k] for k in range(len(words))}
            found_words, found_pos_words = self.analyze_words(concept_set, text)
            found_words_score = len(found_words) / len(concept_set)
            found_pos_score = len(found_pos_words) / len(concept_set)
            found_words_score_metric = 1 if len(found_words) == len(concept_set) else 0
            found_pos_score_metric = 1 if len(found_pos_words) == len(concept_set) else 0
            score = (found_words_score + found_pos_score + natrual_score) / 3
            scores.append(score)
            message = self.get_revise_message(concept_set, found_words, found_pos_words, natrual_score == 1, cs_feedback)
            messages.append(message)
            metrics.append({'cover': found_words_score, 'pos': found_pos_score, 'common_sense': natrual_score, 'cover_metric': found_words_score_metric, 'pos_metric': found_pos_score_metric })
            
        return scores, messages, metrics


class ClusteringCritique(CritiqueBase):
    def __init__(self, device, generator_tokenizer, model_type='llama_2', is_training=False) -> None:
        CritiqueBase.__init__(self, device, generator_tokenizer, model_type, is_training)
        self.students_pattern = r"Group (?P<group_id>(\d)+). (?P<students>.+)."

    def extract_text(self, text, task):
        lines = text.split('\n')
        filtered_lines = []
        for line in lines:
            if line.startswith("Group"):
                filtered_lines.append(line)
            else:
                break
        filtered_text = '\n'.join(filtered_lines)
        if len(filtered_text) == 0:
            return text
        return filtered_text
        
    def batch_critique_impl(self, tasks, texts):
        scores = []
        messages = []
        metrics = []
        
        for text, task in zip (texts, tasks):
            groups_lines = text.strip().split('\n')
            all_students = set(task['students'])
            student_to_group = {}
            constraints = task['constraints']

            mistakes = []
            incorrect_counter = 0
            all_counter = 0
            for i, group_line in enumerate(groups_lines):
                group_line = group_line.strip()
                if len(group_line) == 0:
                    continue
                match = re.search(self.students_pattern, group_line)
                if not match or 'students' not in match.groupdict() or 'group_id' not in match.groupdict():
                    if i > 0:
                        break
                    logger.info(f"Error with output:\n{text}")
                    mistakes.append("The output format is incorrect.")
                    all_counter += 1
                    incorrect_counter += 1
                    break
                
                group_students = match.groupdict()['students']
                group_id = match.groupdict()['group_id']
                students_names = [s.strip().split('.')[0].replace('</s>', '').replace('<|eot_id|>', '').replace('<|end_of_text|>', '') for s in group_students.split(', ')]
                
                if len(students_names) == 1:
                    mistakes.append(f"Group {group_id} has only one student, but each group should consist of at least two students.")
                    incorrect_counter += 1
                    all_counter += 1
                    student_to_group[students_names[0]] = group_id
                    continue
                
                for student in students_names:
                    is_incorrect = 0
                    all_counter += 1
                    if student not in all_students:
                        mistakes.append(f"{student} in group {group_id} is not part of the students list.")
                        incorrect_counter += 1
                        continue
                    
                    if student in student_to_group:
                        new_mistake = f"{student} was grouped multiple times."
                        if new_mistake not in mistakes:
                            mistakes.append(new_mistake)                        
                        is_incorrect = 1
                        
                    student_to_group[student] = group_id
                    for constraint in constraints:
                        first_student, other_student, match = constraint.split('_')
                        if first_student == student:
                            match = True if match == 'True' else False
                            if match and other_student not in students_names:
                                mistakes.append(f"{student} wasn't grouped with {other_student} despite {student}'s request.")
                                is_incorrect = 1
                            elif not match and other_student in students_names:
                                mistakes.append(f"{student} was grouped with {other_student} despite {student}'s request.")
                                is_incorrect = 1
                                
                    incorrect_counter += is_incorrect
            
            for student in all_students:
                if student not in student_to_group:
                    all_counter += 1
                    mistakes.append(f"{student} was not grouped.")   
                    incorrect_counter += 1
            
            score = 1 - incorrect_counter / all_counter
            message = ""
            if len(mistakes) > 0:
                combined_message = ' '.join(mistakes)
                if self.model_type == 'llama_2':
                    message = f"[INST]\n- Feedback: {combined_message} Please Try again. [/INST]\n- Output:"
                elif self.model_type == 'llama_3':
                    message = f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\n- Feedback: {combined_message} Please Try again.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\n- Output:"
            
            scores.append(score)
            messages.append(message)
            metrics.append({'feedback': message})
             

        return scores, messages, metrics

class SentimentCritique(CritiqueBase):
    def __init__(self, device, generator_tokenizer, model_type='llama_2', is_training=False) -> None:
        CritiqueBase.__init__(self, device, generator_tokenizer, model_type, is_training)
        
        self.critique_model = AutoModelForSequenceClassification.from_pretrained("LiYuan/amazon-review-sentiment-analysis").to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained("LiYuan/amazon-review-sentiment-analysis")

    def extract_text(self, text, task):
        text = text.strip()
        if '.' in text:
            index = text.rfind('.')
            text = text[:(index+1)]
        return text
        
    def batch_critique_impl(self, tasks, texts):
        scores = []
        messages = []
        metrics = []
        model_inputs = self.tokenizer(texts, padding='longest', max_length=4096, return_tensors="pt").to(self.critique_model.device)
        with torch.no_grad():
            logits = self.critique_model(**model_inputs).logits

        predicted_class_ids = logits.argmax(dim=1).tolist()
        
        for predicted_class_id, task in zip(predicted_class_ids, tasks):
            predicted_label = self.critique_model.config.id2label[predicted_class_id]
            gold_label = task['label']
            score = 1.0 if predicted_label == gold_label else 0.0 
            scores.append(score)
            message = ""
            if predicted_label != gold_label:
                message = f"Your review classified as {predicted_label} review. Please write a new short {gold_label} review. Do not mention the number of stars."
                message = self.combine_messages([message], "")
            messages.append(message)
            metrics.append({'score': score})

        return scores, messages, metrics
    
class NumPlanningCritique(CritiqueBase):
    def __init__(self, device, generator_tokenizer, model_type='llama_2', is_training=False) -> None:
        CritiqueBase.__init__(self, device, generator_tokenizer, model_type, is_training)
        
    def extract_text(self, text, task):
        if len(text) == 0:
            return text
        if text[0] != ' ':
            text = f" {text}"
        if '.' in text:
            index = text.index('.')
            text = text[:(index+1)]
        return text
    
    def generate_revise_message(self, last_generated_word, generated_count, gold_last_word, gold_word_count, prefix):
        messages = []
        gold_word_count = gold_word_count if isinstance(gold_word_count, int) or isinstance(gold_word_count, float) else gold_word_count.item()
        if generated_count != gold_word_count:
            messages.append(f"You completed the sentence with {generated_count} words. However you should complete the sentence with {gold_word_count} words.")
        if last_generated_word != gold_last_word:
            messages.append(f"The last word should be \"{gold_last_word}\" and not \"{last_generated_word}\".")
                        
        return self.combine_messages(messages, f" {prefix}")
        
    def batch_critique_impl(self, tasks, texts):
        scores = []
        messages = []
        metrics = []
        
        for response, task in zip(texts, tasks):
            gold_last_word = task['last_word']
            N = task['N']
            potenital_words = response.strip().split(' ')
            generated_words = []
            for word in potenital_words:
                word = word.replace('</s>', '').replace('<|eot_id|>', '').replace('<|end_of_text|>', '')
                if any(i.isalnum() for i in word):
                    generated_words.append(word)
            generated_count = len(generated_words)
            if len(generated_words) > 0:
                last_generated_word = generated_words[-1]
            else:
                last_generated_word = ''
            last_generated_word = re.sub(r'\W+', '', last_generated_word)
            
            last_word_score = 1 if gold_last_word == last_generated_word else 0
            gold_count = N if isinstance(N, int) or isinstance(N, float) else N.item()
            word_count_score = 1 if gold_count == generated_count else 0
            reward = 0.8*word_count_score + 0.2*last_word_score

            scores.append(reward)
            message = self.generate_revise_message(last_generated_word, generated_count, gold_last_word, N, task['prefix'])
            messages.append(message)
            metrics.append({'word_count_score': word_count_score, 'last_word_score': last_word_score})

        return scores, messages, metrics


class NumPlanningNoFeedbackCritique(CritiqueBase):
    def __init__(self, device, generator_tokenizer, model_type='llama_2', is_training=False) -> None:
        CritiqueBase.__init__(self, device, generator_tokenizer, model_type, is_training)
        
    def extract_text(self, text, task):
        if len(text) == 0:
            return text
        if text[0] != ' ':
            text = f" {text}"
        if '.' in text:
            index = text.index('.')
            text = text[:(index+1)]
        return text
    
    def generate_revise_message(self, last_generated_word, generated_count, gold_last_word, gold_word_count, prefix):
        messages = []
        gold_word_count = gold_word_count if isinstance(gold_word_count, int) or isinstance(gold_word_count, float) else gold_word_count.item()
        if generated_count != gold_word_count:
            messages.append(f"You completed the sentence with {generated_count} words. However you should complete the sentence with {gold_word_count} words.")
        if last_generated_word != gold_last_word:
            messages.append(f"The last word should be \"{gold_last_word}\" and not \"{last_generated_word}\".")
                        
        final_message = self.combine_messages(messages, f" {prefix}")
        if len(final_message) > 0:
            return self.combine_messages(["Your output is incorrect."], f" {prefix}")
        else:
            return ""
        
    def batch_critique_impl(self, tasks, texts):
        scores = []
        messages = []
        metrics = []
        
        for response, task in zip(texts, tasks):
            gold_last_word = task['last_word']
            N = task['N']
            potenital_words = response.strip().split(' ')
            generated_words = []
            for word in potenital_words:
                word = word.replace('</s>', '').replace('<|eot_id|>', '').replace('<|end_of_text|>', '')
                if any(i.isalnum() for i in word):
                    generated_words.append(word)
            generated_count = len(generated_words)
            if len(generated_words) > 0:
                last_generated_word = generated_words[-1]
            else:
                last_generated_word = ''
            last_generated_word = re.sub(r'\W+', '', last_generated_word)
            
            last_word_score = 1 if gold_last_word == last_generated_word else 0
            gold_count = N if isinstance(N, int) or isinstance(N, float) else N.item()
            word_count_score = 1 if gold_count == generated_count else 0
            reward = 0.8*word_count_score + 0.2*last_word_score

            scores.append(reward)
            message = self.generate_revise_message(last_generated_word, generated_count, gold_last_word, N, task['prefix'])
            messages.append(message)
            metrics.append({'word_count_score': word_count_score, 'last_word_score': last_word_score})

        return scores, messages, metrics

        
class RationaleGenerationCritique(CritiqueBase):
    def __init__(self, device, generator_tokenizer, model_type='llama_2', is_training=False) -> None:
        CritiqueBase.__init__(self, device, generator_tokenizer, model_type, is_training)
        
        model_name = "google/flan-t5-large"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, 
                                                                device_map="balanced_low_0", 
                                                                load_in_8bit=True)
    
    def extract_text(self, text, task):
        text = text.strip()
        sentences = text.split('.')[:2]
        return '.'.join(sentences) + '.'
    
    
    def batch_critique_impl(self, tasks, texts):
        scores = []
        messages = []
        metrics = []
        student_queries = []
        for text, task in zip(texts, tasks):
            student_q_format = task['student_model_query']
            student_query = student_q_format.replace('ADDED_EXP', text)
            student_queries.append(student_query)
            
        student_answers = self.hf_generate(self.model, self.tokenizer, student_queries, max_new_tokens=20)
        
        for student_answer, task in zip(student_answers, tasks):
            gold_answer = task['correct_answer']
            if student_answer in gold_answer:
                score = 1
                message = ""
            else:
                score = 0
                message = "Your background is insufficient - the student picked a wrong answer instead of the correct answer. Revise your background (up to two short sentences) - so the student will succeed. Reminder: Do not mention the correct answer explicitly."
                message = self.combine_messages([message], " Background:")
        
            scores.append(score)
            messages.append(message)
            metrics.append({'score': score})
            
        return scores, messages, metrics
    
class StyleTransferCritique(CritiqueBase):
    def __init__(self, device, generator_tokenizer, model_type='llama_2', is_training=False) -> None:
        CritiqueBase.__init__(self, device, generator_tokenizer, model_type, is_training)
        self.pipe = pipeline(
        task="text-classification",
        model="cffl/bert-base-styleclassification-subjective-neutral"
    )
        
    def extract_text(self, text, task):
        text = text.strip()
        if '.' in text:
            index = text.index('.')
            text = text[:(index+1)]
        return text
    
    
    def batch_critique_impl(self, tasks, texts):
        scores = []
        messages = []
        metrics = []
        
        tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}

        classifications = self.pipe(texts, **tokenizer_kwargs)
        
        for classification, task in zip(classifications, tasks):
            if classification['label'] == 'NEUTRAL':
                score = 1
                message = ""
            else:
                score = 0   
                message = f"Your rephrased sentence is subjective. Sentence: {task['sentence']}"
                message = self.combine_messages([message], "")
            
            scores.append(score)
            messages.append(message)
            metrics.append({'score': score})
            
        return scores, messages, metrics


class PanagramCritique(CritiqueBase):
    def __init__(self, device, generator_tokenizer, model_type='llama_2', is_training=False) -> None:
        CritiqueBase.__init__(self, device, generator_tokenizer, model_type, is_training)
        self.dictionary = enchant.Dict("en_US")
        
    def extract_text(self, text, task):
        text = text.strip()
        if '.' in text:
            index = text.index('.')
            text = text[:index].strip()
        if '\n' in text:
            index = text.index('\n')
            text = text[:index].strip()

        return text.split(' ')[0]
    
    
    def batch_critique_impl(self, tasks, texts):
        scores = []
        messages = []
        metrics = []
                
        for text, task in zip(texts, tasks):
            text = text.replace('</s>', '').replace('<|eot_id|>', '').replace('<|end_of_text|>', '')
            if len(text.split(' ')) > 1 or len(text) == 0:
                scores.append(0)
                messages.append(self.combine_messages(['The output format is incorrect. Please output only the panagram word.'], ' Word:'))
                logger.info(f"Error with output: {text}")
                metrics.append({'valid_score': 0, 'unused_score': 0, 'additional_score': 0})

                continue
            
            task_letters = set(task['letters_list'])
            current_messages = []
            
            valid_score = 1
            unused_score = 1
            additional_score = 1
            
            if not self.dictionary.check(text):
                current_messages.append(f"\"{text}\" is not a valid word in English.")
                valid_score = 0
                                
            used_letters = set(list(text))
            
            unused_letters = task_letters - used_letters
            additional_letters = used_letters - task_letters
            
            
            
            if len(unused_letters) > 0:
                if len(unused_letters) == 1:
                    letter = list(unused_letters)[0]
                    current_messages.append(f"\"{text}\" doesn't have the letter {letter}.")
                else:
                    current_messages.append(f"\"{text}\" doesn't have the letters {', '.join(unused_letters)}.")
                unused_score = 0
            
            if len(additional_letters) > 0:
                if len(additional_letters) == 1:
                    letter = list(additional_letters)[0]
                    current_messages.append(f"\"{text}\" has the letter {letter} which is not on the list.")
                else:
                    current_messages.append(f"\"{text}\" has the letters: {', '.join(additional_letters)} which are not on the list.")
                additional_score = 0
                
            score = (valid_score + unused_score + additional_score) / 3
            message = self.combine_messages(current_messages, " Word:")
            
            scores.append(score)
            messages.append(message)
            metrics.append({'valid_score': valid_score, 'unused_score': unused_score, 'additional_score': additional_score})
            
        return scores, messages, metrics




def run_code(code, queue):
    try:
        exec(code)
    except Exception as e:
        queue.put(e)
            
            
class ExcThread(threading.Thread):

    def __init__(self, bucket, code):
        threading.Thread.__init__(self)
        self.bucket = bucket
        self.code = code

    def run(self):
        try:
            exec(self.code)
        except Exception as e:
            self.bucket.put(sys.exc_info())    
    

def handler(signum, frame):
    raise InfiniteLoopException("end of time")

class InfiniteLoopException(Exception):
    pass

class MbppCritique(CritiqueBase):
    def __init__(self, device, generator_tokenizer, model_type='llama_2', is_training=False) -> None:
        CritiqueBase.__init__(self, device, generator_tokenizer, model_type, is_training)
        
    def extract_text(self, text, task):
        text = text.strip()
        text = text.replace("<|eot_id|>", "")
        text = text.replace('`', '')
        if 'def' in text:
            text = text[text.index('def'):]
        if '\n\n' in text:
            text = text.split('\n\n')[0]

        return text
    
    
    
    
    def batch_critique_impl(self, tasks, texts):
        scores = []
        messages = []
        metrics = []
                
        for text, task in zip(texts, tasks):
            current_messages = []
            score = 1
            compile_score = 1
            function = text
                
            try:
                # https://stackoverflow.com/questions/1463306/how-does-exec-work-with-locals
                local_dict = {}
                byte_code = compile_restricted(function, filename='<inline code>', mode='exec')
                exec(byte_code, {'__builtins__': safe_builtins}, local_dict)
            except Exception as e:
                score = 0
                compile_score = 0
                correctness_score = 0
                current_messages.append("Your function has a syntax error.")

            if compile_score > 0:
                correct_output_count = 0
                tests = task['tests']
                queue = multiprocessing.Queue()

                for test in tests:
                    try:
                        #exec(f'{function}\n\n{test}')
                        code = f'{function}\n\n{test}'
                        p = multiprocessing.Process(target=run_code, args=(code,queue))
                        p.start()

                        # Wait for 10 seconds or until process finishes
                        p.join(10)

                        # If thread is still active
                        if p.is_alive():
                            p.kill()
                            p.join()
                            current_messages.append(f"Your function gets into an infinite loop.")
                            break
                            
                            
                        
                        if not queue.empty():
                            current_messages.append(f"Your function fails on the following unit test:\n\n{test}")
                            break
                        
                    except InfiniteLoopException:
                        current_messages.append(f"Your function gets into an infinite loop.")
                        break
                    except Exception as e:
                        #current_messages.append(f"Your function doesn't pass the unit tests.")
                        current_messages.append(f"Your function fails on the following unit test:\n\n{test}")
                        break

                    correct_output_count = correct_output_count + 1

                correctness_score = correct_output_count / len(tests)                    
                
            score = 0.5*compile_score + 0.5*(correctness_score)
                
        
            message = self.combine_messages(current_messages, "")
            scores.append(score)
            messages.append(message)
            metrics.append({'compile_score': compile_score, 'correctness_score': correctness_score})
            
        return scores, messages, metrics    
    
    
class ProgramCritique(CritiqueBase):
    def __init__(self, device, generator_tokenizer, model_type='llama_2', is_training=False) -> None:
        CritiqueBase.__init__(self, device, generator_tokenizer, model_type, is_training)
        
    def extract_text(self, text, task):
        text = text.strip()
        text = text.replace(":\n", ":", 1)
        text = text.replace("<|eot_id|>", "")
        text = text.split('\n')[0]
        
        return text
    
    
    def batch_critique_impl(self, tasks, texts):
        scores = []
        messages = []
        metrics = []
        LENGTH_MAX_SCORE = 25
        CORRECTNESS_SCORE_MAX = 60
        COMPILE_SCORE = 20
                
        for text, task in zip(texts, tasks):
            current_messages = []
            compile_score = COMPILE_SCORE
            length_score = 0
            correctness_score = 0
                
            try:
                # https://stackoverflow.com/questions/1463306/how-does-exec-work-with-locals
                local_dict = {}
                byte_code = compile_restricted(text, filename='<inline code>', mode='exec')
                exec(byte_code, {'__builtins__': safe_builtins}, local_dict)
                f = local_dict['f']
            except Exception as e:
                compile_score = 0
                score = 0
                current_messages.append("Your function has a syntax error.")

            if compile_score > 0:
                length_score = max(0.0, LENGTH_MAX_SCORE - float(len(text)) / 4)
                if length_score == 0:
                    current_messages.append("Your function is too long.")
                
                correct_output_count = 0
                input_list = task['input_list']
                output_list = task['output_list']
                
                for single_input, expected in zip(input_list, output_list):
                    try:
                        y = f(float(single_input))
                    except Exception:
                        compile_score = 0
                        current_messages.append("Your function has a syntax error.")
                        continue

                    if expected == 'False' or expected == 'True':
                        expected = bool(expected)
                    elif '[' in expected:
                        expected = list(expected)
                    else:
                        try:
                            expected = float(expected)
                        except Exception:
                            expected = expected
                    if y == expected:
                        correct_output_count += 1
                    else:
                        current_messages.append(f"The output of {single_input} is {y}, but it should be {expected}.")
                        
                correctness_score = correct_output_count / len(input_list)                    
                
                sample_score = compile_score + correctness_score * (
                        CORRECTNESS_SCORE_MAX + length_score)
                score = sample_score / (COMPILE_SCORE + CORRECTNESS_SCORE_MAX + LENGTH_MAX_SCORE)
                
                if correct_output_count == len(input_list) and length_score > 0:
                    score = 1
                    current_messages = []
        
            message = self.combine_messages(current_messages, "")
            scores.append(score)
            messages.append(message)
            metrics.append({'compile_score': compile_score, 'length_score': length_score, 'correctness_score': correctness_score})
            
        return scores, messages, metrics
    
class ParaphraseGenerationCritique(CritiqueBase):
    def __init__(self, device, generator_tokenizer, model_type='llama_2', is_training=False) -> None:
        CritiqueBase.__init__(self, device, generator_tokenizer, model_type, is_training)
        
        model_name = "google/flan-t5-large"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="balanced_low_0", load_in_8bit=True)
        self.pipe = pipeline("text-classification", model="AMHR/adversarial-paraphrasing-detector")
        self.run_server()
        self.parser = CoreNLPParser(url=f'http://localhost:{self.port}')
        
    
    def find_free_port(self):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]
    
    def run_server(self):
        self.port = self.find_free_port()
        command1 = subprocess.Popen([f'./apps/stanford-corenlp-full-2018-10-05 ; java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port {self.port}'], shell=True, stdout=subprocess.PIPE)
        time.sleep(10)
        
        
    def reinit_server(self):
        process = Popen(["lsof", "-i", ":{0}".format(self.port)], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        for process in str(stdout.decode("utf-8")).split("\n")[1:]:       
            data = [x for x in process.split(" ") if x != '']
            if (len(data) <= 1):
                continue

            os.kill(int(data[1]), signal.SIGKILL)
        
        time.sleep(5)
        self.run_server()
        self.parser = CoreNLPParser(url=f'http://localhost:{self.port}')
    
    def extract_text(self, text, task):
        text = text.strip()
        if '?' in text:
            index = text.index('?')
            text = text[:(index+1)]
        return text
    
    def generate_revise_message(self, is_paraphrase, ted_score, parse, exemplar, source_sentence):
        messages = []
        if not is_paraphrase:
            messages.append(f"Your question is not a paraphrase of the given question.")
        if ted_score < 1:
            messages.append(f"The syntactic structure of your question is {parse}. Your question should have the following syntactic structure: {exemplar}.")

        if len(messages) > 0:
            return self.combine_messages(messages, "")

        return ""
    
    def batch_critique_impl(self, tasks, texts):
        scores = []
        messages = []
        metrics = []
        
        pipe_inputs = []
        for text, task in zip(texts, tasks):
            pipe_inputs.append(f"{task['source']}{text}")
            
        tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}

        classifications = self.pipe(pipe_inputs, **tokenizer_kwargs)
        
        try:
            pure_parses = get_pure_parse(texts, self.parser)
        except:
            self.reinit_server()
            pure_parses = get_pure_parse(texts, self.parser)
        
        for task, classification, pure_parse in zip(tasks, classifications, pure_parses):
            exemplar = task['exemplar']
            is_paraphrase = classification['label'] == 'LABEL_1'
            ted_e_score, parse = ted_e_3(exemplar, pure_parse)
            paraphrase_score = 1 if is_paraphrase else 0
            ted_score = 1 if ted_e_score == 0 else 0
            reward = 0.5*paraphrase_score + 0.5*ted_score
            scores.append(reward)
            message = self.generate_revise_message(is_paraphrase, ted_score, parse, exemplar, task['source'])
            messages.append(message)
            
            metrics.append({'paraphrase_score': paraphrase_score, 'ted_score': ted_score, 'ted_e_score': ted_e_score})
                
        return scores, messages, metrics
    
class StoryGenerationCritique(CritiqueBase):
    def __init__(self, device, generator_tokenizer, model_type='llama_2', is_training=False) -> None:
        CritiqueBase.__init__(self, device, generator_tokenizer, model_type, is_training)
        model_name= "princeton-nlp/sup-simcse-bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.critique_model = AutoModel.from_pretrained(model_name).to('cuda')
        self.generator_tokenizer = generator_tokenizer
        
    def extract_text(self, text, task):
        return text
    
    def generate_revise_message(self, reptitions, coherence_score, prefix):
        messages = []
        if len(reptitions) > 0:
            repeated_tokens = [self.generator_tokenizer.convert_ids_to_tokens(t).replace('▁', '') for t in reptitions]
            repeated_text = ', '.join(repeated_tokens)
            messages.append(f"You repeated the words: {repeated_text}. Try to make less repeatitions.")
        if coherence_score < 0.3:
            messages.append(f"Your story is not coherent. Rewrite your story to make it more coherent.")
                        
        return self.combine_messages(messages, f" {prefix}")
    
    def batch_critique_impl(self, tasks, texts):
        scores = []
        messages = []
        metrics = []
        batch_tokens = self.generator_tokenizer(texts, padding='longest', max_length=4096)['input_ids']
        
        for text, task, tokens in zip(texts, tasks, batch_tokens):
            prefix = task['prefix']
            rep_4_score, reptitions = rep_n(tokens, 4, self.generator_tokenizer)
            coherence_score = coherence(self.tokenizer, self.critique_model, prefix, text)
            
            rep_final_score = 0 if len(reptitions) > 0 else 1
            coherence_final_score = 0 if coherence_score < 0.3 else 1
            reward = rep_final_score*0.5 + coherence_final_score*0.5

            scores.append(reward)
            message = self.generate_revise_message(reptitions, coherence_score, prefix)
            messages.append(message)
            metrics.append({'rep_score': rep_final_score, 'coherence_score': coherence_final_score, 'rep_4_metric': rep_4_score, 'coherence_metric': coherence_score})

        return scores, messages, metrics
    
        
class MetaLearningCritique(CritiqueBase):
    def __init__(self, device, generator_tokenizer, model_type='llama_2', is_training=False) -> None:
        CritiqueBase.__init__(self, device, model_type, is_training)
        self.task_to_critique = {
            'sentiment': SentimentCritique(device, generator_tokenizer, model_type, is_training),
            'num_planning': NumPlanningCritique(device, generator_tokenizer, model_type, is_training),
            'story_generation': StoryGenerationCritique(device, generator_tokenizer, model_type, is_training),
            'rationale_generation': RationaleGenerationCritique(device, generator_tokenizer, model_type, is_training),
            'paraphrase': ParaphraseGenerationCritique(device, generator_tokenizer, model_type, is_training),
        }
        
    def extract_text(self, text, task):
        return self.task_to_critique[task['task']].extract_text(text, task)
    
    def batch_critique_impl(self, tasks, texts):
        scores = [0 for i in range(len(texts))]
        messages = ["" for i in range(len(texts))]
        metrics = [{} for i in range(len(texts))]
        
        indices_by_tasks = {}
        for i in range(len(texts)):
            task = tasks[i]
            task_name = task['task']
            if task_name not in indices_by_tasks:
                indices_by_tasks[task_name] = []
                
            indices_by_tasks[task_name].append(i)
            
        for task_name, indices_by_task in indices_by_tasks.items():
            current_texts = [texts[idx] for idx in indices_by_task]
            current_tasks = [tasks[idx] for idx in indices_by_task]
            current_scores, current_messages, current_metrics = self.task_to_critique[task_name].batch_critique(current_tasks, current_texts)
            for score, message, metric, idx in zip(current_scores, current_messages, current_metrics, indices_by_task):
                scores[idx] = score
                messages[idx] = message 
                metrics[idx] = metric
                
        return scores, messages, metrics