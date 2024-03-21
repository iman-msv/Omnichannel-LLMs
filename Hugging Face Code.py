# Python Packages
import pandas as pd
import numpy as np
from scrapy import Selector
import re
from langdetect import detect
import emoji
from transformers import pipeline, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import numpy as np
import evaluate

# CSV File
amazon_reviews = pd.read_csv("Amazon Under Armour/Under Armour Men's Tech 2.csv")

# Sentiment Based on Customers Rating
amazon_reviews['rating_sent'] = np.where(amazon_reviews['rating'] < 3, 'negative', 
                                          np.where(amazon_reviews['rating'] == 3, 'neutral', 'positive'))
amazon_reviews['rating_sent'].value_counts()

amazon_reviews['review'] = amazon_reviews['review'].astype(str)

# Sentences Similarity
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
amazon_reviews['embedding'] = amazon_reviews['review'].apply(model.encode)

for i in range(len(amazon_reviews)):
    for j in range(len(amazon_reviews)):
        cosine_scores = util.cos_sim(amazon_reviews['embedding'][i], amazon_reviews['embedding'][j])
        if cosine_scores > 0.6:
            amazon_reviews.iloc[j, 0] = amazon_reviews.iloc[i, 0]

amazon_reviews.drop_duplicates(subset = ['review'], inplace = True)

# Hugging Face
bert_model = 'nlptown/bert-base-multilingual-uncased-sentiment'
bert_classifier = pipeline('text-classification', model=bert_model)

roberta_model = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
roberta_classifier = pipeline('text-classification', model=roberta_model)

distilbert_model = 'lxyuan/distilbert-base-multilingual-cased-sentiments-student'
distilled_student_sentiment_classifier = pipeline(model= distilbert_model, return_all_scores=False)

# Sentiment predictions of each llm
bart_base_preds = bert_classifier(amazon_reviews['review'].to_list())
roberta_base_preds = roberta_classifier(amazon_reviews['review'].to_list())
distilled_student_preds = distilled_student_sentiment_classifier(amazon_reviews['review'].to_list())

amazon_reviews['bert_label_num'] = [int(pred['label'][:1]) for pred in bart_base_preds]
amazon_reviews['bert_label'] = np.where(amazon_reviews['bert_label_num'] < 3, 'negative', 
                                          np.where(amazon_reviews['bert_label_num'] == 3, 'neutral', 'positive'))
amazon_reviews['bert_score'] = [round(pred['score'], 3) for pred in bart_base_preds]


amazon_reviews['roberta_label'] = [pred['label'] for pred in roberta_base_preds]
amazon_reviews['roberta_score'] = [round(pred['score'], 3) for pred in roberta_base_preds]

amazon_reviews['distilbert_label'] = [pred['label'] for pred in distilled_student_preds]
amazon_reviews['distilbert_score'] = [round(pred['score'], 3) for pred in distilled_student_preds]

# Bert Labels
amazon_reviews['bert_label_num'].value_counts()
amazon_reviews['bert_label'].value_counts()

# Roberta Labels
amazon_reviews['roberta_label'].value_counts()

# Distilbert Labels
amazon_reviews['distilbert_label'].value_counts()

# Accuracy and F1 Sentiment Prediction Evaluation
# Bert
f1 = evaluate.load('f1')
accuracy = evaluate.load('accuracy')
references = np.where(amazon_reviews['rating_sent'] == 'negative', 0, 
                      np.where(amazon_reviews['rating_sent'] == 'neutral', 1, 2))
predictions = np.where(amazon_reviews['bert_label'] == 'negative', 0, 
                      np.where(amazon_reviews['bert_label'] == 'neutral', 1, 2))

f1_bert = f1.compute(references = references, predictions = predictions, average = 'weighted')
print(f'F1 score of Bert using weighted average due to imbalace classes is {f1_bert["f1"]:.3f}')

accuracy_bert = accuracy.compute(references = references, predictions = predictions)
print(f'Accuracy score of Bert is {accuracy_bert["accuracy"]:.3f}')

# Roberta
predictions = np.where(amazon_reviews['roberta_label'] == 'negative', 0, 
                      np.where(amazon_reviews['roberta_label'] == 'neutral', 1, 2))

f1_roberta = f1.compute(references = references, predictions = predictions, average = 'weighted')
print(f'F1 score of Roberta using weighted average due to imbalace classes is {f1_roberta["f1"]:.3f}')

accuracy_roberta = accuracy.compute(references = references, predictions = predictions)
print(f'Accuracy score of is {accuracy_roberta["accuracy"]:.3f}')

# Distilbert
predictions = np.where(amazon_reviews['distilbert_label'] == 'negative', 0, 
                      np.where(amazon_reviews['distilbert_label'] == 'neutral', 1, 2))

f1_distilbert = f1.compute(references = references, predictions = predictions, average = 'weighted')
print(f'F1 score of Distilbert using weighted average due to imbalace classes is {f1_distilbert["f1"]:.3f}')

accuracy_distilbert = accuracy.compute(references = references, predictions = predictions)
print(f'Accuracy score of Distilbert is {accuracy_distilbert["accuracy"]:.3f}')

positive_bert = amazon_reviews[amazon_reviews['bert_label'] == 'positive']
negative_bert = amazon_reviews[amazon_reviews['bert_label'] == 'negative']
neutral_bert = amazon_reviews[amazon_reviews['bert_label'] == 'neutral']

# One word reviews can be removed
positive_bert_short = positive_bert[positive_bert['review'].str.split().str.len() > 1]

positive_reviews = ' '.join(positive_bert_short['review'])
negative_reviews = ' '.join(negative_bert['review'])
neutral_reviews = ' '.join(neutral_bert['review'])

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

positive_summarized = summarizer(positive_reviews, max_length = 200)
negative_summarized = summarizer(negative_reviews, max_length = 200)
neutral_summarized = summarizer(neutral_reviews, max_length = 200)

positive_summarized = positive_summarized[0]['summary_text']
negative_summarized = negative_summarized[0]['summary_text']
neutral_summarized = neutral_summarized[0]['summary_text']


print(positive_summarized[0]['summary_text'])
print(negative_summarized[0]['summary_text'])
print(neutral_summarized[0]['summary_text'])

# DialoGPT-medium
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# GPT-2
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
generator(f"You should buy Nike Airmax 2017 becuase:", max_length=100, num_return_sequences=1)



# Fine-tuning
train_dataset = load_dataset('csv', data_files={'train':'amazon_fine_tune_train.csv'})
valid_dataset = load_dataset('csv', data_files={'valid':'amazon_fine_tune_valid.csv'})

# Tokenize dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token = '1')
tokenizer.pad_token = tokenizer.eos_token
tokenized_train_data = train_dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding=True), batched=True)
tokenized_valid_data = valid_dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding=True), batched=True)