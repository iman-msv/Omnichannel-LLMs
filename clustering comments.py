# Packages
import pandas as pd
import numpy as np
import re
from langdetect import detect
import emoji
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
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
    path = f"Amazon Nike Men's Air Max 2017/Amazon Nike Men's Air Max P{i}.html"
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

# Create a condition indicating whether the review contains enough characters
more_one_char = amazon_comments['review'].apply(lambda x: len(x) > 1)

# Filtering reviews
amazon_comments = amazon_comments[more_one_char]

# Removing reviews that only contain emojies
is_emoji = amazon_comments['review'].apply(emoji.purely_emoji)
amazon_comments = amazon_comments[~is_emoji]

# Detecting Language of Review
amazon_comments['language'] = amazon_comments['review'].apply(lambda x: detect(x))

# Reviews in a language other than English
amazon_comments[amazon_comments['language'] != 'en']

# Filtering espanish
amazon_comments = amazon_comments[amazon_comments['language'] != 'es']

amazon_comments.drop(columns=['language'], inplace=True)

# Dropping N/A from reivews
amazon_comments = amazon_comments[amazon_comments['review'] != 'N/A']

# Reset Index
amazon_comments.reset_index(drop=True, inplace=True)

# Inspecting Data
amazon_comments.head()
amazon_comments.info()
amazon_comments.nunique()

# Location is constant
amazon_comments.drop(columns=['location'], inplace=True)

# Days since first review
first_rev_data = amazon_comments['date'].min()
amazon_comments['first_review_days'] = (amazon_comments['date'] - first_rev_data).dt.days

# Exploratory Data Analysis
sns.set_theme(style="whitegrid")
sns.countplot(x = 'rating', data = amazon_comments)
plt.show()

# Days Since First Review
sns.histplot(x = 'first_review_days', data = amazon_comments)
plt.show()

# Replacing u with you
# Replacing informal spelling
informal_spell = {
    r'\bu\b': 'you',
    r'\byr\b': 'year'
}

amazon_comments['review'].replace(informal_spell, regex=True, inplace=True)

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

# Without removing stopwords (for sentinemt)
def lemma_withstopwords(sentences):
    doc = nlp(sentences)
    lemmas = [token.lemma_ if (token.text.isupper() and token.text != "I") else token.lemma_.lower() for token in doc]
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
def clustering_func(data, cluster_method, k = None, hi_method = 'ward', scale = True, transform = True):
    # Skewness Reduction
    if transform:
        data = PowerTransformer(method='yeo-johnson').fit_transform(data)

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


# Dimension Reduction Function
def dim_reduc(data, method, n_components = None, scale = True, transform = True):
    # Skewness Reduction
    if transform:
        data = PowerTransformer(method='yeo-johnson').fit_transform(data)
    
    # Scale 
    if scale:
        data = StandardScaler().fit_transform(data)
    
    if method == 'PCA':
        if n_components == None:
        # Perform PCA
            pca = PCA()
            pca.fit_transform(data)
            # Calculate cumulative proportion of variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_proportion = np.cumsum(explained_variance)
            print('Explained Variance:', explained_variance)
            print('Cumulative Proportion:', cumulative_proportion)

        else:
            pca = PCA(n_components = n_components)
            pca_comp = pca.fit_transform(data)
            return pca_comp
    
    if method == 'SVD':
        if n_components == None:
        # Perform SVD
            svd = TruncatedSVD()
            svd.fit_transform(data)
            # Calculate cumulative proportion of variance
            explained_variance = svd.explained_variance_ratio_
            cumulative_proportion = np.cumsum(explained_variance)
            print('Explained Variance:', explained_variance)
            print('Cumulative Proportion:', cumulative_proportion)

        else:
            svd = TruncatedSVD(n_components = n_components)
            svd_comp = svd.fit_transform(data)
            return svd_comp


# Word Embedding
# Doc2Vec
def dic2vec(reviews_df, column_name, vector_size):
    # Tokenize the reviews
    tokenized_reviews = [word_tokenize(review.lower()) for review in reviews_df[column_name]]

    # Tag the tokenized reviews
    tagged_reviews = [TaggedDocument(words=review_words, tags=[str(i)]) for i, review_words in enumerate(tokenized_reviews)]

    # Train a Doc2Vec model
    model = Doc2Vec(tagged_reviews, vector_size=vector_size, window=2, min_count=2, workers=4, epochs=100)

    # Get a vector for each review
    review_vectors = [model.dv[str(i)] for i in range(len(reviews_df))]

    # Convert the list of vectors to a DataFrame
    vectors_df = pd.DataFrame(review_vectors)

    return vectors_df

dic2vec_reviews = dic2vec(amazon_comments, 'review_cleaned_nostop', vector_size=30)
dic2vec_reviews.columns = [f'vec{i}' for i in range(dic2vec_reviews.shape[1])]

clustering_func(data = dic2vec_reviews, cluster_method = 'hierarchical')
clustering_func(data = dic2vec_reviews, cluster_method = 'kmeans')

# Suggest 3 Clusters
amazon_comments['kmeans_doc2vec_4clust'] = clustering_func(data = dic2vec_reviews, cluster_method = 'kmeans', k = 4)

# Number of Cases in Each Cluster
amazon_comments['kmeans_doc2vec_4clust'].value_counts()

# Dimension Reduction
dim_reduc(dic2vec_reviews, method='PCA')

# 4 Componenets Cover 90% of Information
dic2vec_reduced_reviews = dim_reduc(dic2vec_reviews, method='PCA', n_components=4)
dic2vec_reduced_reviews = pd.DataFrame(dic2vec_reduced_reviews)
dic2vec_reduced_reviews.columns = [f'pr{i}' for i in range(dic2vec_reduced_reviews.shape[1])]

# Clustering After PCA
clustering_func(data = dic2vec_reduced_reviews, cluster_method = 'hierarchical')
clustering_func(data = dic2vec_reduced_reviews, cluster_method = 'kmeans')

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
clustering_func(data = amazon_comments[['polarity', 'subjectivity', 'rating']], cluster_method = 'hierarchical', k = 3)
clustering_func(data = amazon_comments[['polarity', 'subjectivity', 'rating']], cluster_method = 'kmeans')

# Suggesting 4 Clusters 
amazon_comments['kmeans_sent_4clust'] = clustering_func(data = amazon_comments[['polarity', 'subjectivity', 'rating']], cluster_method = 'kmeans', k = 4)

# Combined Features
combined_features = amazon_comments[['polarity', 'subjectivity', 'rating']].merge(dic2vec_reviews, left_index=True, right_index=True)

# Clustering with Combined Features
clustering_func(combined_features, cluster_method='hierarchical')
clustering_func(combined_features, cluster_method='kmeans')

# Suggesting 3 Clusters 
amazon_comments['kmeans_comb_3clust'] = clustering_func(data = combined_features, cluster_method = 'kmeans', k = 3)

# Comparing Clusters
def comparing_clusts(clusters):
    # Comparing Clusters
    print(amazon_comments[clusters].value_counts())
    print(amazon_comments.groupby(clusters)['rating'].value_counts(normalize=True))
    print(amazon_comments.groupby(clusters)['polarity'].mean())

    sns.displot(x = 'polarity', data = amazon_comments, kind='kde', hue = clusters)
    plt.show()

    sns.boxplot(x = clusters, y = 'polarity', data = amazon_comments)
    plt.show()

    sns.displot(x = 'subjectivity', data = amazon_comments, kind='kde', hue = clusters)
    plt.show()

    sns.displot(x = 'num_words', data = amazon_comments, kind='kde', hue = clusters)
    plt.show()

comparing_clusts('kmeans_doc2vec_4clust')
comparing_clusts('kmeans_sent_4clust')
comparing_clusts('kmeans_comb_3clust')

# Words in each segment
# Conditions
group1 = amazon_comments['kmeans_sent_4clust'] == 0
group2 = amazon_comments['kmeans_sent_4clust'] == 1
group3 = amazon_comments['kmeans_sent_4clust'] == 2
group4 = amazon_comments['kmeans_sent_4clust'] == 3

# Unigram Group 1
unigrams = word_count_func(amazon_comments[group1], (1,1))
sns.barplot(x = 'count', y = 'word', data = unigrams[:10])
plt.show()

# Unigram Group 2
unigrams = word_count_func(amazon_comments[group2], (1,1))
sns.barplot(x = 'count', y = 'word', data = unigrams[:10])
plt.show()

# Unigram Group 3
unigrams = word_count_func(amazon_comments[group3], (1,1))
sns.barplot(x = 'count', y = 'word', data = unigrams[:10])
plt.show()

# Unigram Group 4
unigrams = word_count_func(amazon_comments[group4], (1,1))
sns.barplot(x = 'count', y = 'word', data = unigrams[:10])
plt.show()

# Bigrams Group 1
bigrams = word_count_func(amazon_comments[group1], (2,2))
sns.barplot(x = 'count', y = 'word', data = bigrams[:10])
plt.show()

# Bigrams Group 2
bigrams = word_count_func(amazon_comments[group2], (2,2))
sns.barplot(x = 'count', y = 'word', data = bigrams[:10])
plt.show()

# Bigrams Group 3
bigrams = word_count_func(amazon_comments[group3], (2,2))
sns.barplot(x = 'count', y = 'word', data = bigrams[:10])
plt.show()

# Bigrams Group 4
bigrams = word_count_func(amazon_comments[group4], (2,2))
sns.barplot(x = 'count', y = 'word', data = bigrams[:10])
plt.show()

# Similarity
amazon_comments['spacy_doc'] = amazon_comments['review'].apply(nlp)

# Compute a similarity matrix
similarity_matrix = amazon_comments['spacy_doc'].apply(lambda doc1: amazon_comments['spacy_doc'].apply(lambda doc2: doc1.similarity(doc2)))

# Sorting by Clusters
sorted_amazon = amazon_comments[['review', 'kmeans_sent_4clust', 'polarity', 'rating']].sort_values('kmeans_sent_4clust')
sorted_amazon.loc[:, 'review'] = [f'{i}. ' for i in range(1, len(sorted_amazon) + 1)] + sorted_amazon.loc[:, 'review']

# Write XSLX
sorted_amazon.to_excel("Amazon Nike Men's Air Max 2017/Airmax Clustered Reviews.xlsx")

# Rating and Polarity Statistics
sorted_amazon.groupby('kmeans_sent_4clust')[['polarity', 'rating']].agg(['mean', 'std'])