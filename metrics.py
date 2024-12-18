from scipy.spatial.distance import cosine
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

STOPWORDS = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
             'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
             'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
             'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were',
             'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to',
             'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have',
             'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can',
             'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
             'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by',
             'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'also', 'could', 'would'}


def rep_n(tokens, n, tokenizer):
    rep_counter = 0
    counter = 0
    special_tokens = tokenizer.all_special_ids
    repeated_tokens = set()
    for i in range(len(tokens) - n):
        current_token = tokens[i + n]
        token = tokenizer.convert_ids_to_tokens(current_token)
        if token == None or current_token in special_tokens or token[0] != '▁' or not any(letter.isalpha() for letter in token) or len(token) == 2:
            continue
        if token.replace('▁', '') in STOPWORDS:
            continue
        if current_token in tokens[i:i+n]:
            rep_counter += 1
            repeated_tokens.add(current_token)
        counter += 1
            
    avg = rep_counter / (counter + 1e-8)
    return avg, repeated_tokens

def coherence(tokenizer, model, prefix, suffix):
    texts = [
        prefix,
        suffix
    ]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(model.device)

    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    # Calculate cosine similarities
    cosine_sim = 1 - cosine(embeddings[0].to('cpu'), embeddings[1].to('cpu'))
    return cosine_sim