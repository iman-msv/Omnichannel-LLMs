# Packages
import pandas as pd

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

