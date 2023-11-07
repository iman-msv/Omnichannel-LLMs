# Packages
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
import spacy

# Import CSV file
'''
The prompt we entered to generate this file is as follows:
Suppose you are responsible for writing ad messages in a fashion retailer that sells 
clothes with affordable prices and acceptable quality. The main purpose at this stage is brand awareness. 
The messages are going to be used in billboards around the city. 
Generate 50 ad messages without writing anything further:
'''
chatgpt = pd.read_csv('chatgpt responses.csv')

# First few rows 
print(chatgpt.head())

# Remove quatation marks
def discard_quatation_marks(sentence):
    """Discards " from a sentence.

    Args:
    sentence: A string representing a sentence.

    Returns:
    A string representing the sentence without quotation marks.
    """

    sentence = sentence[1:-1]
    return(sentence)

chatgpt = chatgpt.applymap(discard_quatation_marks)


# Changing column names
chatgpt.columns = ['billboard', 'website', 'salesman']

# Counting words in chatgpt dataframe 
def str_len(x):
    return x.str.len()

chatgpt_counts = chatgpt.apply(str_len)

# Comparing counts
print(chatgpt_counts.describe())

# Evidently, messages generated for salesman are longer

# Frequent words in each column
joined_billboard_words = ' '.join(chatgpt['billboard'])

billboard_count = Counter(word_tokenize(joined_billboard_words))

print(sorted(billboard_count.items(), key= lambda x: x[1], reverse=True))

# Text preprocessing
billboard_tokens = [token for token in word_tokenize(joined_billboard_words.lower()) if token.isalpha()]
billboard_no_stops = [nos_token for nos_token in billboard_tokens if nos_token not in stopwords.words('english')]

# Wordcloud
cloud_billboard = WordCloud().generate(' '.join(billboard_no_stops))
plt.imshow(cloud_billboard, interpolation='bilinear')
plt.show()

# Function for Word frequency
def word_count(sentences):
    joined_list = ' '.join(sentences)
    tokens = [token for token in word_tokenize(joined_list.lower()) if token.isalpha()]
    no_stops = [nos_token for nos_token in tokens if nos_token not in stopwords.words('english')]
    count = Counter(no_stops)
    print(sorted(count.items(), key= lambda x: x[1], reverse=True))
    cloud_generate = WordCloud().generate(' '.join(no_stops))
    plt.imshow(cloud_generate, interpolation='bilinear')
    plt.show()

word_count(chatgpt['billboard'])
word_count(chatgpt['website'])
word_count(chatgpt['salesman'])

# Tf-idf
# TFIDF Function 
def tfidf_func(sentences):
    # Tokenizing
    tokens = [word_tokenize(doc) for doc in sentences.str.lower().to_list()]
    no_stops = [nos_token for nos_token in tokens if nos_token not in stopwords.words('english')]
    # Gensim Dict
    dict = Dictionary(no_stops)
    # Corpus Creation
    corpus = [dict.doc2bow(doc) for doc in tokens]
    #Instantiate TFIDF
    tfidf = TfidfModel(corpus)
    # Words with Maximum TFIDF
    max_tfidfs = []
    for i in range(len(corpus)):
        max_tfidfs.append(max(tfidf[corpus[i]], key = lambda x: x[1]))

    # Sorting with respect to TFIDF
    sorted_tfidfs = sorted(max_tfidfs,  key = lambda x: x[1], reverse=True)

    # ID to Token
    token_id_to_token = {token_id: token for token, token_id in dict.token2id.items()}
    tokens_with_tfidf = [(token_id_to_token[token_id], tfidf_value) for token_id, tfidf_value in sorted_tfidfs]
    
    return tokens_with_tfidf

tfidf_func(chatgpt['billboard'])
tfidf_func(chatgpt['website'])
tfidf_func(chatgpt['salesman'])

# Similarity
nlp = spacy.load("en_core_web_sm")
def calculate_column_similarity(column1, column2):
    doc1 = nlp(" ".join(column1))
    doc2 = nlp(" ".join(column2))
    return doc1.similarity(doc2)

calculate_column_similarity(chatgpt['billboard'], chatgpt['website'])
calculate_column_similarity(chatgpt['billboard'], chatgpt['salesman'])
calculate_column_similarity(chatgpt['website'], chatgpt['salesman'])