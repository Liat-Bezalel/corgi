import json
import re
from nltk.tree import Tree
from zss import simple_distance, Node

def ted_e_3(gold_parse, pure_parse):
    predicted_parse_pruned = trim_str(pure_parse, 3)
    return compute_tree_edit_distance(predicted_parse_pruned, gold_parse), predicted_parse_pruned
    
def convert_str(string):
    new_list= []
    for ele in string.split(" "):
        if ")" in ele:
            new_list.append(str(re.sub(r'^.*?\)', ')', ele)))
        else:
            new_list.append(ele)
    new_str = " ".join(ele for ele in new_list)
    return new_str

# def get_pure_parse(sentence, parser):
#     output = nlp.annotate(sentence, properties={
#     'annotators': 'parse',
#     'outputFormat': 'json'
#     })
#     output = json.loads(output)
#     parse = output['sentences'][0]['parse'].split('\n')
#     pure_parse = ''
#     for p in parse:
#         new_p = convert_str(p)
#         pure_parse += ' ' + new_p.strip()
#     return pure_parse

def get_pure_parse(sentences, parser):
    trees = list(parser.raw_parse_sents(sentences))
    pure_parses = []                
    for tree in list(trees):
        tree = list(tree)[0]
        parse = convert_str(str(tree))

        pure_parse = ''
        for p in parse.split('\n'):
            new_p = convert_str(p)
            pure_parse += ' ' + new_p.strip()
        pure_parses.append(pure_parse.strip())
    return pure_parses
    
def strdist(a, b):
    if a == b:
        return 0
    else:
        return 1

def build_tree(s):
    old_t = Tree.fromstring(s)
    new_t = Node("S")

    def create_tree(curr_t, t):
        if t.label() and t.label() != "S":
            new_t = Node(t.label())
            curr_t.addkid(new_t)
        else:
            new_t = curr_t
        for i in t:
            if isinstance(i, Tree):
                create_tree(new_t, i)
    create_tree(new_t, old_t)
    return new_t

def string_comma(string):
    start = 0
    new_string = ''
    while start < len(string):
        if string[start:].find(",") == -1:
            new_string += string[start:]
            break
        else:
            index = string[start:].find(",")
            if string[start - 2] != "(":
                new_string += string[start:start + index]
                new_string += " "
            else:
                new_string = new_string[:start-1] +", "
            start = start + index + 1
    return new_string

def clean_tuple_str(tuple_str):
    new_str_ls = []
    if len(tuple_str) == 1:
        new_str_ls.append(tuple_str[0])
    else:
        for i in str(tuple_str).split(", "):
            if i.count("'") == 2:
                new_str_ls.append(i.replace("'", ""))
            elif i.count("'") == 1:
                new_str_ls.append(i.replace("\"", ""))
    str_join = ' '.join(ele for ele in new_str_ls)
    return string_comma(str_join)

def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)

def trim_tree_nltk(root, height):
    try:
        root.label()
    except AttributeError:
        return

    if height < 1:
        return
    all_child_state = []
    #     print(root.label())
    all_child_state.append(root.label())

    if len(root) >= 1:
        for child_index in range(len(root)):
            child = root[child_index]
            if trim_tree_nltk(child, height - 1):
                all_child_state.append(trim_tree_nltk(child, height - 1))
    #                 print(all_child_state)
    return all_child_state


def trim_str(string, height):
    return clean_tuple_str(to_tuple(trim_tree_nltk(Tree.fromstring(string), height)))
    
def compute_tree_edit_distance(pred_parse, ref_parse):
    return simple_distance(build_tree(ref_parse), build_tree(pred_parse), label_dist=strdist)