# Packages
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

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
tokens = [token for token in word_tokenize(joined_billboard_words.lower()) if token.isalpha()]
no_stops = [nos_token for nos_token in tokens if nos_token not in stopwords.words('english')]

# Wordcloud
cloud_billboard = WordCloud().generate(' '.join(no_stops))
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