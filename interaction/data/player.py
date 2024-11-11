import os
import random
import json

from .invalid import get_invalid_words,get_invalied_ids

def save_json(obj, save_dir, file_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir,file_name)
    with open(save_path,'w') as f:
        json.dump(obj,f,indent=4,ensure_ascii=False)
    return True

def choose_player_words(sentences, player_num, save_dir = None, file_name = 'player_words'):
    """
    choose the words of the player

    params
    =======
    sentences: list
            the list of input sentences [str1,str2,str3,...]
    player_num: int
            the num of the player words in one sentences, commomly, player_num<=10
    save_dir: str
            the save dictory to save the player words, in form of json file.

    return 
    =======
    player_words: list
            the list of the player words [[word1,word2,...],...]
    """
    invalid_words = get_invalid_words()
    player_words = []
    for sentence in sentences:
        player_words_sen = [word for word in sentence.split(' ') if word not in invalid_words and word != '']
        sample_index = random.sample([i for i in range(len(player_words_sen))],player_num)
        sample_index.sort()
        player_words_sen = [player_words_sen[idx] for idx in sample_index]
        player_words.append(player_words_sen)

    if save_dir is None:
        return player_words
    
    save_json(player_words,save_dir,f'{file_name}.json')
    return player_words


def get_player_ids_from_word(tokenizer, sentences, player_words, with_indent, save_dir=None, file_name='player_ids_from_word'):
    """
    Based on the player words, and the given sentences, get the player's token postions in sentencs

    params
    =======
    tokenizer:
            tokenizer
    sentences: list
            the list of input sentences [str1,str2,str3,...]
    player_words: int
            the list of the player words [[word1,word2,...],...]
    save_dir: str
            the save dictory to save the player words, in form of json file.

    return 
    =======
    player_words: list
            the list of the player words [[word1,word2,...],...]
    """
    assert len(sentences) == len(player_words)
    player_ids = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        player_words_sen = player_words[i]
        input_ids_sen = tokenizer.encode(sentence)
        
        player_ids_sen = []
        pointer = 0
        for word in player_words_sen:
            # if with indent, add space in front of the word
            if with_indent:
                word = ' ' + word

            word_ids = tokenizer.encode(word,add_special_tokens=False)
            while True:
                if pointer == len(input_ids_sen):
                    raise RuntimeError('do not find coresdpongding words')
                if input_ids_sen[pointer:pointer + len(word_ids)] == word_ids:
                    player_ids_sen.append([pos for pos in range(pointer,pointer + len(word_ids))])
                    pointer += len(word_ids)
                    break
                pointer += 1
        player_ids.append(player_ids_sen)   

    if  save_dir is None:
        return player_ids
            
    save_json(player_ids,save_dir,f'{file_name}.json')
    return player_ids


def get_player_ids_from_token(tokenizer, sentences, save_dir=None, file_name="player_ids_from_token"):
    """
    Based on the tokenizer, get the player's token postions in sentencs, each player is a token

    params
    =======
    tokenizer:
            tokenizer
    sentences: list
            the list of input sentences [str1,str2,str3,...]
    save_dir: str
            the save dictory to save the player words, in form of json file.

    return 
    =======
    player_words: list
            the list of the player words [[word1,word2,...],...]
    """
    invalied_token_ids = get_invalied_ids(tokenizer)
    player_ids = []
    for sentence in sentences:
        input_ids_sen = tokenizer.encode(sentence)
        player_ids_sen = [[pos] for pos in range(len(input_ids_sen)) if input_ids_sen[pos] not in invalied_token_ids]
        player_ids.append(player_ids_sen)

    if  save_dir is None:
        return player_ids

    save_json(player_ids,save_dir,f'{file_name}.json')
    return player_ids


if __name__ == '__main__':
    pass
