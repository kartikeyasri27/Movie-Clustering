import numpy as np
import pandas as pd
import nltk
from sklearn import feature_extraction

from variables import *
from tokenizer import *


# Building Vocabulary
# ===================

totalvocab_stemmed = []
totalvocab_tokenized = []

for i in synopses:
    allwords_stemmed = tokenize_and_stem(i) 
    totalvocab_stemmed.extend(allwords_stemmed) 
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)


# K Means Clustering
# ==================


# Vectorizing synopses
# --------------------
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=2_00_000, min_df=0.2, stop_words='english', use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)
terms = tfidf_vectorizer.get_feature_names()

# Finding Cosine Similarity
# -------------------------
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

# Model Building
# --------------
from sklearn.cluster import KMeans
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

# Saving to disk
# --------------
from sklearn.externals import joblib
joblib.dump(km, 'doc_cluster.pkl')
km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

films = { 'title': titles, 'rank': ranks, 'synopsis': synopses, 'cluster': clusters, 'genre': genres }
frame = pd.DataFrame(films, index = [clusters] , columns = ['rank', 'title', 'cluster', 'genre'])
grouped = frame['rank'].groupby(frame['cluster']) 
grouped.mean()

# Multi Dimensional Scaling
# -------------------------
from sklearn.manifold import MDS
MDS()
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)
xs, ys = pos[:, 0], pos[:, 1]

# Visualizing Output 
# ------------------
import matplotlib.pyplot as plt
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
cluster_names = {0: 'Marries, Leaves, Tells', 1: 'Family, Home, War', 2: 'Murders, Tells, Police', 3: 'Army, Soldier, Command', 4: 'Father, New York, Fight'}

df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 

groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(40, 20))
ax.margins(0.05)

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis= 'x', which='both', bottom='off', top='off', labelbottom='off')
    ax.tick_params(axis= 'y', which='both', left='off', top='off', labelleft='off')

ax.legend(numpoints=1)

for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

plt.savefig('clusters_small_noaxes.png', dpi=200)
# plt.show()


# Heirarchical Clustering
# =======================


# Model Building
# --------------
from scipy.cluster.hierarchy import ward
linkage_matrix = ward(dist)

# Visualizing Output
# ------------------
from scipy.cluster.hierarchy import dendrogram

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=titles);

plt.tick_params(axis= 'x', which='both', bottom='off', top='off', labelbottom='off')
plt.tight_layout()
plt.savefig('ward_clusters.png', dpi=200)


# Latent Dirichlet Allocation Clustering
# ======================================


# Preprocessing
# -------------
import string
from nltk.tag import pos_tag

def strip_proppers(text):
    tagged = pos_tag(text.split())
    non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return " ".join(non_propernouns)

preprocess = [strip_proppers(doc) for doc in synopses]
tokenized_text = [tokenize_and_stem(text) for text in preprocess]
texts = [[word for word in text if word not in nltk.corpus.stopwords.words('english')] for text in tokenized_text]

from gensim import corpora, models, similarities 
dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=1, no_above=0.8)
corpus = [dictionary.doc2bow(text) for text in texts]

# Model Building
# --------------
lda = models.LdaModel(corpus, num_topics=5, id2word=dictionary, update_every=5, chunksize=10000, passes=100)

# Display Output
# --------------
topics_matrix = lda.show_topics(formatted=False, num_words=20)
topic_words = []

for i in range(0,len(topics_matrix)):
    topics = []
    for topic in topics_matrix[i][1]:
        topics.append(topic[0])
    topic_words.append(topics)

for i in topic_words:
    print([str(word) for word in i])
    print()

