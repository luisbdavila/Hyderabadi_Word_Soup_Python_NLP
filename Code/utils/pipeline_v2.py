import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_mutual_info_score

from collections import defaultdict, Counter
from tqdm import tqdm
from unidecode import unidecode
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.base import BaseEstimator
# Note: This is the pipeline_v2 file used during class with some very minor changes to accomodate for our needs.


class MainPipeline(BaseEstimator):
    def __init__(self,
                 print_output = False,
                 no_emojis = True,
                 no_hashtags = True,
                 hashtag_retain_words = True,
                 no_newlines = True,
                 no_html_tags = True,
                 no_urls = True,
                 no_punctuation = True,
                 no_stopwords = True,
                 custom_stopwords = [],
                 ignored_stopwords = [],
                 convert_diacritics = True,
                 lowercase = True,
                 lemmatized = True,
                 list_pos = ["n","v","a","r","s"],
                 pos_tags_list = "no_pos",
                 tokenized_output = False):

        self.print_output = print_output
        self.no_emojis = no_emojis
        self.no_hashtags = no_hashtags
        self.hashtag_retain_words = hashtag_retain_words
        self.no_html_tags = no_html_tags
        self.no_urls = no_urls
        self.no_punctuation = no_punctuation
        self.no_stopwords = no_stopwords
        self.custom_stopwords = custom_stopwords
        self.ignored_stopwords = ignored_stopwords
        self.convert_diacritics = convert_diacritics
        self.lowercase = lowercase
        self.lemmatized = lemmatized
        self.list_pos = list_pos
        self.pos_tags_list = pos_tags_list
        self.tokenized_output = tokenized_output

    def regex_cleaner(self, raw_text):

        #patterns
        html_tag_pattern = "(\\[a-z])"
        hashtags_at_pattern = "([#\@@\u0040\uFF20\uFE6B])"
        hashtags_ats_and_word_pattern = "([#@]\w+)"
        emojis_pattern = "([\u2600-\u27FF])"
        url_pattern = "(?:\w+:\/{2})?(?:www)?(?:\.)?([a-z\d]+)(?:\.)([a-z\d\.]{2,})(\/[a-zA-Z\/\d]+)?" ##Note that this URL pattern is *even better*
        punctuation_pattern = "[\u0021-\u0026\u0028-\u002C\u002E-\u002F\u003A-\u003F\u005B-\u005F\u007C\u2010-\u2028\ufeff`]+"
        apostrophe_pattern = "'(?=[A-Z\s])|(?<=[a-z\.\?\!\,\s])'"
        separated_words_pattern = "(?<=\w\s)([A-Z]\s){2,}"
        ##Note that this punctuation_pattern doesn't capture ' this time to allow our tokenizer to separate "don't" into ["do", "n't"]

        if self.no_emojis == True:
            clean_text = re.sub(emojis_pattern,"",raw_text)
        else:
            clean_text = raw_text

        if self.no_hashtags == True:
            if self.hashtag_retain_words == True:
                clean_text = re.sub(hashtags_at_pattern,"",clean_text)
            else:
                clean_text = re.sub(hashtags_ats_and_word_pattern,"",clean_text)

        if self.no_html_tags == True:
            clean_text = re.sub(html_tag_pattern," ",clean_text)
            
        if self.no_urls == True:
            clean_text = re.sub(url_pattern,"",clean_text)
        
        if self.no_punctuation == True:
            clean_text = re.sub(punctuation_pattern," ",clean_text)
            clean_text = re.sub(apostrophe_pattern," ",clean_text)

        return clean_text

    def lemmatize_all(self, token):
    
        wordnet_lem = nltk.stem.WordNetLemmatizer()
        for arg_1 in self.list_pos[0]:
            token = wordnet_lem.lemmatize(token, arg_1)
        return token

    def main_pipeline(self, raw_text):
        
        """Preprocess strings according to the parameters"""
        if self.print_output == True:
            print("Preprocessing the following input: \n>> {}".format(raw_text))

        clean_text = self.regex_cleaner(raw_text)

        if self.print_output == True:
            print("Regex cleaner returned the following: \n>> {}".format(clean_text))

        tokenized_text = nltk.tokenize.word_tokenize(clean_text)

        tokenized_text = [re.sub("'m","am",token) for token in tokenized_text]
        tokenized_text = [re.sub("n't","not",token) for token in tokenized_text]
        tokenized_text = [re.sub("'s","is",token) for token in tokenized_text]

        if self.no_stopwords == True:
            stopwords = nltk.corpus.stopwords.words("english")
            tokenized_text = [item for item in tokenized_text if item.lower() not in stopwords or
                              item.lower() in self.ignored_stopwords] #### ISTO
        
        if self.convert_diacritics == True:
            tokenized_text = [unidecode(token) for token in tokenized_text]

        if self.lemmatized == True:
            tokenized_text = [self.lemmatize_all(token) for token in tokenized_text]
    
        if self.no_stopwords == True:
            tokenized_text = [item for item in tokenized_text if item.lower() not in self.custom_stopwords or
                              item.lower() in self.ignored_stopwords]

        if self.pos_tags_list == "pos_list" or self.pos_tags_list == "pos_tuples" or self.pos_tags_list == "pos_dictionary":
            pos_tuples = nltk.tag.pos_tag(tokenized_text)
            pos_tags = [pos[1] for pos in pos_tuples]
        
        if self.lowercase == True:
            tokenized_text = [item.lower() for item in tokenized_text]
        
        if self.pos_tags_list == "pos_list":
            return (tokenized_text, pos_tags)
        elif self.pos_tags_list == "pos_tuples":
            return pos_tuples   
        
        else:
            if self.tokenized_output == True:
                return tokenized_text
            else:
                detokenizer = TreebankWordDetokenizer()
                detokens = detokenizer.detokenize(tokenized_text)
                if self.print_output == True:
                    print("Pipeline returning the following result: \n>> {}".format(str(detokens)))
                return str(detokens)
        

def cooccurrence_matrix_sentence_generator(preproc_sentences, sentence_cooc=False, window_size=5):

    co_occurrences = defaultdict(Counter)

    # Compute co-occurrences
    if sentence_cooc == True:
        for sentence in tqdm(preproc_sentences):
            for token_1 in sentence:
                for token_2 in sentence:
                    if token_1 != token_2:
                        co_occurrences[token_1][token_2] += 1
    else:
        for sentence in tqdm(preproc_sentences):
            for i, word in enumerate(sentence):
                for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                    if i != j:
                        co_occurrences[word][sentence[j]] += 1

    #ensure that words are unique
    unique_words = list(set([word for sentence in preproc_sentences for word in sentence]))

    # Initialize the co-occurrence matrix
    co_matrix = np.zeros((len(unique_words), len(unique_words)), dtype=int)

    # Populate the co-occurrence matrix
    word_index = {word: idx for idx, word in enumerate(unique_words)}
    for word, neighbors in co_occurrences.items():
        for neighbor, count in neighbors.items():
            co_matrix[word_index[word]][word_index[neighbor]] = count

    # Create a DataFrame for better readability
    co_matrix_df = pd.DataFrame(co_matrix, index=unique_words, columns=unique_words)

    co_matrix_df = co_matrix_df.reindex(co_matrix_df.sum().sort_values(ascending=False).index, axis=1)
    co_matrix_df = co_matrix_df.reindex(co_matrix_df.sum().sort_values(ascending=False).index, axis=0)

    # Return the co-occurrence matrix
    return co_matrix_df

def word_freq_calculator(td_matrix, word_list, df_output=True):
    word_counts = np.sum(td_matrix, axis=0).tolist()
    if df_output == False:
        word_counts_dict = dict(zip(word_list, word_counts))
        return word_counts_dict
    else:
        word_counts_df = pd.DataFrame({"words":word_list, "frequency":word_counts})
        word_counts_df = word_counts_df.sort_values(by=["frequency"], ascending=False)
        return word_counts_df
    
def plot_term_frequency(df, nr_terms, df_name, show=True):
    
    # Create the Seaborn bar plot
    plt.figure(figsize=(10, 8))
    sns_plot = sns.barplot(x='frequency', y='words', data=df.head(nr_terms))  # Plotting top 20 terms for better visualization
    plt.title('Top 20 Term Frequencies of {}'.format(df_name))
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    if show==True:
        plt.show()

    fig = sns_plot.get_figure()
    plt.close()

    return fig

def replace_words_dict(word_list, reversed_dict):
    return [reversed_dict.get(word, word) for word in word_list]


def cooccurrence_network_generator(cooccurrence_matrix_df, n_highest_words, output=None):
    
    filtered_df = cooccurrence_matrix_df.iloc[:n_highest_words, :n_highest_words]
    graph = nx.Graph()

    # Add nodes for words and set their sizes based on frequency
    for word in filtered_df.columns:
        graph.add_node(word, size=filtered_df[word].sum())

    # Add weighted edges to the graph based on co-occurrence frequency
    for word1 in filtered_df.columns:
        for word2 in filtered_df.columns:
            if word1 != word2:
                graph.add_edge(word1, word2, weight=filtered_df.loc[word1, word2])

    figure = plt.figure(figsize=(20, 20))

    # Generate positions for the nodes
    pos = nx.spring_layout(graph, k=0.5)

    # Calculate edge widths based on co-occurrence frequency
    edge_weights = [0.1 * graph[u][v]['weight'] for u, v in graph.edges()]

    # Get node sizes based on the frequency of words
    node_sizes = [data['size'] * 2 for _, data in graph.nodes(data=True)]

    # Create the network graph
    nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=node_sizes)
    nx.draw_networkx_edges(graph, pos, edge_color='gray', width=edge_weights)
    nx.draw_networkx_labels(graph, pos, font_weight='bold', font_size=8)

    plt.show() 

    if output=="return":
        return figure
    
def elbow_finder(tf_matrix, max_k=10, verbose=True):
    
    y_inertia = []
    for k in tqdm(range(1,max_k+1)):
        kmeans = KMeans(n_clusters=k,random_state=0).fit(tf_matrix)
        if verbose==True:
            print("For k = {}, inertia = {}".format(k,round(kmeans.inertia_,3)))
        y_inertia.append(kmeans.inertia_)

    x = np.array([1, max_k])
    y = np.array([y_inertia[0], y_inertia[-1]])
    coefficients = np.polyfit(x, y, 1)
    line = np.poly1d(coefficients)

    a = coefficients[0]
    c = coefficients[1]

    elbow_point = max(range(1, max_k+1), key=lambda i: abs(y_inertia[i-1] - line(i)) / np.sqrt(a**2 + 1))
    print(f'Optimal value of k according to the elbow method: {elbow_point}')
    
    return elbow_point

def cluster_namer(dataset, label_column_name, nr_words=5):

    labels = list(set(dataset[label_column_name]))
    # corpus generator
    corpus = []
    for label in labels:
        label_doc = ""
        for doc in dataset["dishes_correct_string"].loc[dataset[label_column_name]==label]:        
            label_doc = label_doc + " " + doc
        corpus.append(label_doc)
        

    bow_vectorizer = CountVectorizer(ngram_range=(1,1), token_pattern=r"(?u)\b\w+\b")
    label_name_list = []

    for idx, document in enumerate(corpus):
        corpus_bow_td_matrix = bow_vectorizer.fit_transform(corpus)
        corpus_bow_word_list = bow_vectorizer.get_feature_names_out()

        label_vocabulary = word_freq_calculator(corpus_bow_td_matrix[idx].toarray(),\
                                                                        corpus_bow_word_list, df_output=True)
        
        label_vocabulary = label_vocabulary.head(nr_words)
        label_name = ""
        for jdx in range(len(label_vocabulary)):
            label_name = label_name + "_" + label_vocabulary["words"].iloc[jdx]

        label_name_list.append(label_name)

    label_name_dict = dict(zip(labels,label_name_list))
    dataset[label_column_name] = dataset[label_column_name].map(lambda label : label_name_dict[label])

    return dataset


def plotter_3d_cluster(dataset_org, vector_column_name, cluster_label_name, write_html=False, html_name="test.html"):
    
    dataset = dataset_org.copy()
    dataset = cluster_namer(dataset, cluster_label_name)

    svd_n3 = TruncatedSVD(n_components=3)
    td_matrix = np.array([[component for component in doc] for doc in dataset[vector_column_name]])
    svd_result = svd_n3.fit_transform(td_matrix)

    for component in range(3):
        col_name = "svd_d3_x{}".format(component)
        dataset[col_name] = svd_result[:,component].tolist()

    fig = px.scatter_3d(dataset,
                        x='svd_d3_x0',
                        y='svd_d3_x1',
                        z='svd_d3_x2',
                        color=cluster_label_name,
                        title=vector_column_name+"__"+cluster_label_name,
                        opacity=0.7,
                        hover_name = "dishes_correct_string",
                        color_discrete_sequence=px.colors.qualitative.Alphabet)

    if write_html==True:
        fig.write_html(html_name)
    fig.show()

def plot_dendrogram(model, **kwargs) -> None:
    '''
    Create linkage matrix and then plot the dendrogram

    Arguments:
        ----------
         - model(HierarchicalClustering Model): hierarchical clustering model.
         - **kwargs

    Returns:
        ----------
         None, but a dendrogram is produced.
    '''
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def unsupervised_score_calculator(model_list):
    for tuple in model_list:
        #Inertia
        if tuple[0] not in ["hdbscan", "optics"]:
            print("Inertia of {}: {}".format(tuple[0],tuple[1].inertia_))
        #Silhouette Score
        print("Silhouette score of {}: {}".format(tuple[0],silhouette_score(tuple[2],tuple[1].labels_)))
        #calinski-harabasz
        print("Calinski-Harabasz score of {}: {}".format(tuple[0],calinski_harabasz_score(tuple[2],tuple[1].labels_)))
        print("\n")