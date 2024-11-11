import os
import string

def get_words(words_path):
    """
    Given the word path, return the word list, or sentence list

    params
    =======
    words_path: str
            the txt file to save the word

    return 
    =======
    words: list
            the list of the words [str1,str2,str3,...]
    """
    with open(words_path,'r') as f:
        words = f.readlines()
    words = [word.rstrip() for word in words]
    return words

def get_ids(ids_path):
    """
    Given the ids path, return the ids list

    params
    =======
    ids_path: str
            the txt file to save the ids

    return 
    =======
    ids: list
            the list of the word ids [int1,int2,int3,...]
    """
    with open(ids_path,"r") as f:
        ids = f.readlines()
    ids = [eval(idx.rstrip()) for idx in ids]
    return ids

def get_invalid_words():
    """
    Get the invalid words

    params
    =======

    return 
    =======
    invalid_words: list
            the list of the invalid words [str1,str2,str3,...]
    """
    data_dir = os.path.dirname(__file__)
    stopwords_path = os.path.join(data_dir,'stopwords/english')
    special_word_path = os.path.join(data_dir,'special_char/special_words.txt')
    invalid_words = set()
    invalid_words =invalid_words.union(set(get_words(stopwords_path)))
    invalid_words =invalid_words.union(set(get_words(special_word_path)))
    invalid_words =invalid_words.union(set(string.punctuation))
    
    return invalid_words

def get_invalied_ids(tokenizer):
    """
    Based on the tokenizer, get the invalid token ids

    params
    =======
    tokenizer:
            tokenizer

    return 
    =======
    invalid_token_ids: set
            the set of the invalid token ids (int1,int2,...)
    """
    data_dir = os.path.dirname(__file__)
    special_id_path = os.path.join(data_dir,'special_char/special_ids.txt')
    invalid_token_ids = get_ids(special_id_path)

    invalid_words = get_invalid_words()
    for word in invalid_words:
        invalid_token_ids = invalid_token_ids + tokenizer.encode(word, add_special_tokens=False)

    return set(invalid_token_ids)