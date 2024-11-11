import string

# stopwords consists of 179 English words, which are frequently used and have little meaning,"the","a","an","in"...etc.

stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've", "you'll",
               "you'd", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "she's",
               "her", "hers", "herself", "it", "it's", "its", "itself", "they", "them", "their", "theirs",
               "themselves", "what", "which", "who", "whom", "this", "that", "that'll", "these", "those",
               "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
               "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
               "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
               "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
               "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where",
               "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
               "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
               "don", "don't", "should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
               "aren't", "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn",
               "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't",
               "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't",
               "won", "won't", "wouldn", "wouldn't"}

stop_words = stop_words.union(set(string.punctuation))
stop_words = stop_words.union(set(string.digits)) # 0 to 9
stop_words = stop_words.union({str(num) for num in range(10, 21)}) # 10 to 20
stop_words = stop_words.union(set(string.ascii_letters)) # single lower/upper-case letters
stop_words = stop_words.union({"th",})
stop_words = stop_words.union(set([f"device='cuda:{num}')" for num in range(0,10)]))