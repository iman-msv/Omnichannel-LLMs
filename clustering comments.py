# Packages
import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import scipy.cluster as sc
from spacytextblob.spacytextblob import SpacyTextBlob
from scrapy import Selector
import matplotlib.pyplot as plt
import seaborn as sns

# Loading HTMLs
# Number of Pages
num_pages = 10

def scraping_review_pages_onebyone(html_file_path):
    with open(html_file_path, 'r') as file:
        html_string = file.read()

    sel = Selector(text=html_string)
    for review in sel.xpath('//div[contains(@id, "customer_review")]'):
        yield {
            'review': review.xpath('./div[4]/span/span/text()').get() or np.nan,
            'title': review.xpath('./div[2]/a/span[2]/text()').get(),
            'rating': review.xpath('./div[2]/a/i/span/text()').get(),
            'location_date': review.xpath('./span/text()').get(),
            'purchased_product':review.xpath('./div[3]/a/text()').extract() or np.nan
        }

    
reviews = []

for i in range(1, num_pages + 1):
    path = f'Amazon Customer Review P{i}.html'
    reviews.extend(scraping_review_pages_onebyone(path))

# Creating DataFrame
amazon_comments = pd.DataFrame(reviews)

amazon_comments.info()

# Filter out Not Verified Purchases
amazon_comments.dropna(inplace=True)

# Rating Must Be Float
amazon_comments['rating'] = amazon_comments['rating'].apply(lambda x: x[:3]).astype('float16')

# Parcing Date and Location. Dates Must Be in Date Type
amazon_comments['date'] = amazon_comments['location_date'].apply(lambda x: re.findall('Reviewed in (.*) on (.*)', x)[0][1])
amazon_comments['date'] = pd.to_datetime(amazon_comments['date'])
amazon_comments['location'] = amazon_comments['location_date'].apply(lambda x: re.findall('Reviewed in (.*) on (.*)', x)[0][0])
amazon_comments.drop(columns=['location_date'], inplace=True)

# Parsing Product Features
# Size
amazon_comments['size'] = amazon_comments['purchased_product'].apply(lambda x: re.findall('Size: (\d+)', x[0])[0]).astype('float16')

# Color
amazon_comments['color'] = amazon_comments['purchased_product'].apply(lambda x: re.findall('Color: (.*)', x[1])[0])

# Dropping Original Column
amazon_comments.drop(columns = ['purchased_product'], inplace=True)

# Reset Index
amazon_comments.reset_index(drop=True, inplace=True)

# Inspecting Data
amazon_comments.head()
amazon_comments.info()
amazon_comments.nunique()

# Exploratory Data Analysis
sns.set_theme(style="whitegrid")
sns.countplot(x = 'rating', data = amazon_comments)
plt.show()

# Replacing u with you
amazon_comments['review'] = amazon_comments['review'].str.replace(r'\bu\b', 'you', regex=True)

# Replacing I"m with I'm (one sentence has this issue)
amazon_comments['review'] = amazon_comments['review'].str.replace('I"m', "I'm")

# Spacy
nlp = spacy.load('en_core_web_md')

# Stopwords
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

def lemma_nostopwords(sentences):
    doc = nlp(sentences)
    lemmas = [token.lemma_.lower() for token in doc]
    lemmas = [lemma for lemma in lemmas if lemma.isalpha() and lemma not in spacy_stopwords]
    pre_processed = ' '.join(lemmas)
    return(pre_processed)

amazon_comments['review_cleaned_nostop'] = amazon_comments['review'].apply(lemma_nostopwords)

# Without removing stopwords
def lemma_withstopwords(sentences):
    doc = nlp(sentences)
    lemmas = [token.lemma_.lower() for token in doc]
    lemmas = [lemma for lemma in lemmas if (lemma.isalpha() or lemma == '!')]
    pre_processed = ' '.join(lemmas)
    return(pre_processed)

amazon_comments['review_cleaned_withstop'] = amazon_comments['review'].apply(lemma_withstopwords)

# Feature: Len of sentences
amazon_comments['num_words'] = amazon_comments['review_cleaned_nostop'].apply(lambda x: len(x.split(' ')))

# Number of Words Histogram
sns.histplot(x = 'num_words', data = amazon_comments, binwidth=5)
plt.xlabel('Number of Words in a Review')
plt.show()

# Counting Words
def word_count_func(data, ngrams):
    # Instantiate Vectorizer
    vectorizer = CountVectorizer(ngram_range=ngrams)
    # Bag of Words
    bag_of_words = vectorizer.fit_transform(data['review_cleaned_nostop'])
    # Word Label
    feature_names = vectorizer.get_feature_names_out()
    # Bag of Words Dataframe
    df_bow = pd.DataFrame(bag_of_words.toarray(), columns=feature_names)

    word_counts_df = df_bow.sum(axis=0).sort_values(ascending=False).reset_index()
    word_counts_df.columns = ['word', 'count']

    return(word_counts_df)

# Unigram
unigrams = word_count_func(amazon_comments, (1,1))
sns.barplot(x = 'count', y = 'word', data = unigrams[:10])
plt.show()

# Bigrams
bigrams = word_count_func(amazon_comments, (2,2))
sns.barplot(x = 'count', y = 'word', data = bigrams[:10])
plt.show()

# Trigrams
trigrams = word_count_func(amazon_comments, (3,3))
sns.barplot(x = 'count', y = 'word', data = trigrams[:10])
plt.show()

# Clustering Function
def clustering_func(data, cluster_method, k = None, hi_method = 'ward', scale = False):

    # StandardScaler
    if scale:
        data = StandardScaler().fit_transform(data)

    if cluster_method == 'hierarchical':
        # Choosing Appropriate K
        hierar_link = sc.hierarchy.linkage(data, method=hi_method, metric='euclidean')
        sc.hierarchy.dendrogram(hierar_link)
        plt.show()

        # Return Cluster Labels
        if k != None:
            return sc.hierarchy.fcluster(hierar_link, t=k, criterion='maxclust')

    if cluster_method == 'kmeans':
        # Choosing Appropriate K
        distortions = []
        num_clusters = range(2, 8)
        silhouette_avg = []

        for i in num_clusters:
            centroids, distortion = sc.vq.kmeans(data, k_or_guess=i, iter=100, seed=1)
            distortions.append(distortion)
            clusters, _ = sc.vq.vq(data, centroids)
            sil_avg = silhouette_score(data, labels=clusters)
            silhouette_avg.append(sil_avg)

        elbow_plot_df = pd.DataFrame({
            'num_clusters': num_clusters,
            'distortions': distortions,
            'silhouette_avg': silhouette_avg
        })

        sns.pointplot(x = 'num_clusters', y = 'distortions', data = elbow_plot_df)
        plt.show()

        sns.pointplot(x = 'num_clusters', y = 'silhouette_avg', data = elbow_plot_df)
        plt.show()

        if k != None:
            centroids, distortion = sc.vq.kmeans(data, k_or_guess=k, iter=100, seed=1)
            labels, _ = sc.vq.vq(data, centroids)
        
            # Return Labels
            return labels

# Clustering with BOW
vectorizer = CountVectorizer(ngram_range=(1,1))
bag_of_words = vectorizer.fit_transform(amazon_comments['review_cleaned_nostop'])
bow_df = pd.DataFrame(bag_of_words.toarray(), index=amazon_comments.index, columns=vectorizer.get_feature_names_out())

# Dimension Reduction
svd = TruncatedSVD(n_components=2)
svd.fit(bag_of_words.toarray())
svd.explained_variance_

X_reduced = svd.transform(bag_of_words.toarray())
reduced_df = pd.DataFrame(X_reduced)

# Clustering BOW (unigram)
clustering_func(data = reduced_df, cluster_method = 'hierarchical')
clustering_func(data = reduced_df, cluster_method = 'kmeans')

# Suggesting 3 Clusters
amazon_comments['kmeans_unigram_3clust'] = clustering_func(data = reduced_df, cluster_method = 'kmeans', k = 3)

# Clustering BOW (bigram)
vectorizer = CountVectorizer(ngram_range=(2,2))
bag_of_words = vectorizer.fit_transform(amazon_comments['review_cleaned_nostop'])
bow_df = pd.DataFrame(bag_of_words.toarray(), index=amazon_comments.index, columns=vectorizer.get_feature_names_out())

# Word Embedding
# Doc2Vec
def dic2vec(reviews_df, column_name):
    # Tokenize the reviews
    tokenized_reviews = [word_tokenize(review.lower()) for review in reviews_df[column_name]]

    # Tag the tokenized reviews
    tagged_reviews = [TaggedDocument(words=review_words, tags=[str(i)]) for i, review_words in enumerate(tokenized_reviews)]

    # Train a Doc2Vec model
    model = Doc2Vec(tagged_reviews, vector_size=10, window=2, min_count=2, workers=4, epochs=100)

    # Get a vector for each review
    review_vectors = [model.dv[str(i)] for i in range(len(reviews_df))]

    # Convert the list of vectors to a DataFrame
    vectors_df = pd.DataFrame(review_vectors)

    return vectors_df

dic2vec_reviews = dic2vec(amazon_comments, 'review_cleaned_nostop')

clustering_func(data = dic2vec_reviews, cluster_method = 'hierarchical')
clustering_func(data = dic2vec_reviews, cluster_method = 'kmeans')

# Suggest 2 Clusters
amazon_comments['kmeans_doc2vec_2clust'] = clustering_func(data = dic2vec_reviews, cluster_method = 'kmeans', k = 2)

# Feature: Sentiment Score
nlp_added = spacy.load('en_core_web_md')
nlp_added.add_pipe('spacytextblob')

def sent_func(document):
    doc = nlp_added(document)
    return pd.Series([doc._.blob.polarity, doc._.blob.subjectivity])

amazon_comments[['polarity', 'subjectivity']] = amazon_comments['review_cleaned_withstop'].apply(sent_func)

# Sentiment Histograms
sns.histplot(x = 'polarity', data = amazon_comments)
plt.show()

sns.histplot(x = 'subjectivity', data = amazon_comments)
plt.show()

# Polarity vs. Rating
sns.boxplot(x = 'rating', y = 'polarity', data = amazon_comments)
plt.show()

# Clustering with Sentiment
clustering_func(data = amazon_comments[['polarity', 'subjectivity', 'rating']], cluster_method = 'hierarchical')
clustering_func(data = amazon_comments[['polarity', 'subjectivity', 'rating']], cluster_method = 'kmeans')

# Suggesting 3 or 4 Clusters 
amazon_comments['hierar_sent_3clust'] = clustering_func(data = amazon_comments[['polarity', 'subjectivity', 'rating']], cluster_method = 'hierarchical', k = 3)

# Comparing Clusters
amazon_comments['hierar_sent_3clust'].value_counts()
amazon_comments.groupby('hierar_sent_3clust')['rating'].value_counts(normalize=True)
amazon_comments.groupby('hierar_sent_3clust')['polarity'].mean()

sns.displot(x = 'polarity', data = amazon_comments, kind='kde', hue = '')
plt.show()

sns.boxplot(x = 'hierar_sent_3clust', y = 'polarity', data = amazon_comments)
plt.show()

sns.displot(x = 'subjectivity', data = amazon_comments, kind='kde', hue = 'hierar_sent_3clust')
plt.show()

sns.displot(x = 'num_words', data = amazon_comments, kind='kde', hue = 'hierar_sent_3clust')
plt.show()

# Clustering with Several Features
# 'polarity', 'subjectivity', 'rating', 'dic2vec_reviews'
clustering_df = amazon_comments[['polarity', 'subjectivity', 'rating']]
clustering_df.merge(dic2vec_reviews, left_index=True, right_index=True)

clustering_func(clustering_df, cluster_method='hierarchical', scale=True)
clustering_func(clustering_df, cluster_method='kmeans', scale=True)

# Suggesting 4 Clusters
amazon_comments['kmeans_4clust'] = clustering_func(clustering_df, cluster_method='kmeans', scale=True, k = 4)

amazon_comments['kmeans_4clust'].value_counts()
amazon_comments.groupby('kmeans_4clust')['rating'].value_counts(normalize=True)
amazon_comments.groupby('kmeans_4clust')['polarity'].mean()
amazon_comments.groupby('kmeans_4clust')['num_words'].mean()

sns.displot(x = 'polarity', data = amazon_comments, kind='kde', hue = 'kmeans_4clust')
plt.show()

sns.boxplot(x = 'kmeans_4clust', y = 'polarity', data = amazon_comments)
plt.show()

sns.displot(x = 'subjectivity', data = amazon_comments, kind='kde', hue = 'kmeans_4clust')
plt.show()

# Replacing Labesl
amazon_comments['kmeans_4clust'].replace({
    0:'Subjective Short Satisfied',
    1:'Dissatisfied',
    2:'Long Satisfied',
    3:'Neutral'
}, inplace=True)


# Common Words in Each Segment
subj_short_sat = amazon_comments[amazon_comments['kmeans_4clust'] == 'Subjective Short Satisfied']
dissat = amazon_comments[amazon_comments['kmeans_4clust'] == 'Dissatisfied']
long_satisfied = amazon_comments[amazon_comments['kmeans_4clust'] == 'Long Satisfied']
neutral = amazon_comments[amazon_comments['kmeans_4clust'] == 'Neutral']

# Unigrams
word_count_func(subj_short_sat, (1,1))[:10]
word_count_func(dissat, (1,1))[:10]
word_count_func(neutral, (1,1))[:10]
word_count_func(long_satisfied, (1,1))[:10]

# Bigrams
word_count_func(subj_short_sat, (2,2))[:10]
word_count_func(dissat, (2,2))[:10]
word_count_func(neutral, (2,2))[:10]
word_count_func(long_satisfied, (2,2))[:10]