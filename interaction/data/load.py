import json

def load_sentences(
        data_path: str,
        with_indent: bool):
    """
    Given the data path, load the sentences

    Parameters
    ----------
    data_path: str
    with_indent: bool
        if add indent to the sentences

    Return
    ----------
    sentences: list[sentence_1,sentence_2, ... sentence_N]
                list of player word id, index begins with 0,
    """
    assert data_path.endswith('.txt')
    with open(data_path,'r') as f:
        sentences = f.readlines()
        if with_indent:
            sentences = [' ' + sentence.strip() for sentence in sentences]
        else:
            sentences = [sentence.strip() for sentence in sentences]
    return sentences

def load_player_words(data_path):
    """
    Given the data path, get the player words.

    Parameters
    ----------
    data_path: str

    Return
    ----------
    player_ids: list[[int,int,...,int],[int,int],...]
                list of unencoded sentences
    """
    assert data_path.endswith('.json')
    with open(data_path,'r') as f:
        player_words = json.load(f)
    return player_words

def load_player_ids(data_path):
    """
    Given the data path, get the player indexes.

    Parameters
    ----------
    data_path: str

    Return
    ----------
    player_ids: list[[int,int,...,int],[int,int],...]
                list of unencoded sentences
    """
    assert data_path.endswith('.json')
    with open(data_path,'r') as f:
        player_ids = json.load(f)
    return player_ids
