import collections
import csv
import json
import os
import re
import time

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from langdetect import detect
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize.casual import TweetTokenizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from collections import Counter
from html import unescape

from readProjects.api.scratchApi import get_all_comments_for_project
import seaborn as sns

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')
stops = stopwords.words('english')
urlRegularExpression = re.compile("https:\/\/scratch\.mit\.edu\S+ ", re.IGNORECASE)
OUT_FOLDER = "C:/Users/Isabella/Documents/Masterthesis/Code/out/tmp"

def analyze_comments(projects):
    words = []
    word_counts = []
    word_counts_en = []
    word_counts_en_no_stop = []
    nr_projects_with_comments = 0
    for project in projects:
        project_words = project.get_comment_words()
        if len(project_words) != 0:
            words.extend(project_words)
        project_word_counts, project_word_counts_en, project_word_counts_en_no_stop = project.get_comment_word_counts()
        word_counts.extend(project_word_counts)
        word_counts_en.extend(project_word_counts_en)
        word_counts_en_no_stop.extend(project_word_counts_en_no_stop)
        if len(project_word_counts) != 0:
            nr_projects_with_comments = nr_projects_with_comments + 1
    print(word_counts_en_no_stop)
    get_comment_statistics(word_counts, word_counts_en, nr_projects_with_comments)
    fdist = FreqDist(words)
    fdist.most_common(20)
    fdist.plot(20, cumulative=False)
    plt.savefig("WordFreqDist.png")
    plt.savefig("WordFreqDist.pdf")
    plt.show()
    fdist.tabulate()
    return word_counts_en_no_stop


def get_comment_statistics(word_counts, word_counts_en, nr_projects_with_comments):
    print(len(word_counts))
    print(len(word_counts_en))
    print("# Comments / # Projects")
    print(len(word_counts) / nr_projects_with_comments)
    print("# English Comments / # Total Comments")
    print(len(word_counts_en) / len(word_counts))
    print("# Average_word_numbers")
    print(sum(word_counts) / len(word_counts))
    print("# Average_word_numbers_en")
    print(sum(word_counts_en) / len(word_counts_en))


def get_list_of_words(projects):
    word_list = []
    for project in projects:
        word_list.extend(project.get_comment_words())
    return word_list


def get_tfidf_vectorizer(projects_out_path):
    documents = read_description_and_title_from_projects(projects_out_path)

    # print(documents)
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(2, 3))  # ngram_range=(3,3), using trigrams, use TfidfTransformer if alcrady used CountVectorizer
    vectorizer.fit(documents)
    # print(vectorizer.vocabulary_)
    # print(vectorizer.idf_)
    vector = vectorizer.transform(documents)
    return vectorizer, vector, documents


def get_tfidf_features(projects_out_path):
    vectorizer, vector, documents = get_tfidf_vectorizer(projects_out_path)
    # encode document
    # vector.toarray()
    # print(vector.shape)  # sparse array
    # print(vector)  # final scoring of each word, normalized 0-1
    svd = TruncatedSVD(
        n_components=75)  # reduce dimensionality/factorisation of matrix with singular value decompostion
    svdMatrix = svd.fit_transform(vector)
    # print(svdMatrix)
    terms = vectorizer.get_feature_names()
    print(len(terms))
    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
    # df = pd.DataFrame(columns=['x', 'y', 'document'])
    # df['x'], df['y'], df['document'] = svdMatrix[:,0], svdMatrix[:,1], range(len(vector))
    # plt.scatter()
    # plt.show()
    color = []
    color_map = dict()
    for i, comp in enumerate(svd.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:7]
        print("Topic " + str(i) + ": ")
        topics = []
        for t in sorted_terms:
            color_map[t[0]] = i
            topics.append(t[0])
        print(', '.join(topics))

    tsne_lsa_model = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=2000, verbose=1, random_state=0,
                          angle=0.75)
    tsne_lsa_vectors = tsne_lsa_model.fit_transform(svdMatrix)
    print(tsne_lsa_vectors)
    sns.scatterplot(x=tsne_lsa_vectors[:, 0], y=tsne_lsa_vectors[:, 1], legend="full", alpha=0.3,
                    palette=sns.color_palette("hls", 10))
    y = [0, 1]
    category_to_color = {0: 'darkgreen', 1: 'limegreen'}
    category_to_label = {0: 'A', 1: 'B'}

    fig, ax = plt.subplots(1, 1)
    for category, color in category_to_color.items():
        mask = y == category
        ax.plot(tsne_lsa_vectors[mask, 0], tsne_lsa_vectors[mask, 1], 'o', color=color,
                label=category_to_label[category])
    ax.legend(loc='best')
    plt.show()



    # embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(svdMatrix)
    # plt.figure(figsize=7)
    # dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
    # print(dataset.target)
    # print(svd.components_)
    # color = [sns.color_palette()[x] for x in svd.components_.map(color_map)]
    # print(color)
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=sns.color_palette(), s=10, edgecolors='none')
    # plt.show()

    lda_model = LatentDirichletAllocation(n_components=10, learning_method='online', random_state=0, verbose=0)
    lda_topic_matrix = lda_model.fit_transform(vector)  # eig. doc-term-matrix
    tsne_lda_model = TSNE(
        TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=2000, verbose=1, random_state=0, angle=0.75))
    tsne_lda_vectors = tsne_lda_model.fit_transform(lda_topic_matrix)
    print(tsne_lda_vectors)


def read_all_comments_from_projects(projects_out_path, comment_file):
    texts = []
    project_folders = __get_project_folders(projects_out_path)
    all_comments_path = os.path.join(OUT_FOLDER, 'all_metainfo_' + comment_file + '.json')
    if os.path.isfile(all_comments_path):
        with open(all_comments_path) as all_comments_file:
            return json.load(all_comments_file)
    else:
        for idx, project_folder in enumerate(project_folders):
            comments = read_all_comments_from_project(os.path.join(projects_out_path, project_folder), comment_file, [],
                                                      WordNetLemmatizer())
            if len(comments) > 0:
                text = ' '.join(comments)
                texts.append(text)
            if idx % 10000 == 0:
                print("read_all_comments" + comment_file + ": " + str(idx))
        with open(all_comments_path, 'w') as all_comments_file:
            json.dump(texts, all_comments_file)
        return texts


def read_description_and_title_from_projects(projects_out_path):
    texts = []
    all_titles_path = os.path.join(OUT_FOLDER, 'all_metainfo.json')
    if os.path.isfile(all_titles_path):
        with open(all_titles_path) as all_titles_file:
            return json.load(all_titles_file)
    else:
        project_folders = __get_project_folders(projects_out_path)
        for idx, project_folder in enumerate(project_folders):
            text = ' '.join(read_title_from_project(os.path.join(projects_out_path, project_folder)))
            texts.append(text)
            if idx % 10000 == 0:
                print("read_all_titles: " + str(idx))
        with open(all_titles_path, 'w') as all_titles_file:
            json.dump(texts, all_titles_file)
        return texts


def read_title_from_project(project_folder):
    all_words = []
    project_folder_dir_list = os.listdir(project_folder)
    if 'project_metainfo.json' not in project_folder_dir_list:
        print(project_folder)
        return all_words
    lemmatizer = WordNetLemmatizer()
    project_metainfo_file_path = os.path.join(project_folder, 'project_metainfo.json')
    with open(project_metainfo_file_path) as project_metainfo_file:
        project_metainfo = json.load(project_metainfo_file)
        pos_list_metainfo = []
        all_words.extend(preprocess_string(project_metainfo.get('title'), pos_list_metainfo, lemmatizer))
        all_words.extend(preprocess_string(project_metainfo.get('description'), pos_list_metainfo, lemmatizer))
        all_words.extend(preprocess_string(project_metainfo.get('instructions'), pos_list_metainfo, lemmatizer))
        return all_words


def read_all_strings_from_projects(projects_out_path):
    texts = []
    all_strings_tmp_path = os.path.join(OUT_FOLDER, 'all_strings.json')
    if os.path.isfile(all_strings_tmp_path):
        with open(all_strings_tmp_path) as all_strings_tmp_file:
            return json.load(all_strings_tmp_file)
    else:
        project_folders = __get_project_folders(projects_out_path)
        for idx, project_folder in enumerate(project_folders):
            text = ' '.join(read_all_strings_from_project(os.path.join(projects_out_path, project_folder)))
            texts.append(text)
            if idx % 10000 == 0:
                print("read_all_strings done: " + str(idx))
        with open(all_strings_tmp_path, 'w') as all_strings_tmp_file:
            json.dump(texts, all_strings_tmp_file)
        return texts


def read_all_strings_from_project(project_folder):
    all_words = []
    project_folder_dir_list = os.listdir(project_folder)
    if 'project_metainfo.json' not in project_folder_dir_list:
        print(project_folder)
        return all_words
    elif 'pos.json' in project_folder_dir_list:
        with open(os.path.join(project_folder, 'pos.json')) as pos_file:
            pos_dict = json.load(pos_file)
            all_words.extend([word for (word, pos_tag) in pos_dict['pos_metainfo']])
            all_words.extend([word for (word, pos_tag) in pos_dict['pos_project_comments']])
            all_words.extend([word for (word, pos_tag) in pos_dict['pos_code_comments']])
            return all_words

    lemmatizer = WordNetLemmatizer()
    project_metainfo_file_path = os.path.join(project_folder, 'project_metainfo.json')
    with open(project_metainfo_file_path) as project_metainfo_file:
        project_metainfo = json.load(project_metainfo_file)
        pos_list_metainfo = []
        all_words.extend(preprocess_string(project_metainfo.get('title'), pos_list_metainfo, lemmatizer))
        all_words.extend(preprocess_string(project_metainfo.get('description'), pos_list_metainfo, lemmatizer))
        all_words.extend(preprocess_string(project_metainfo.get('instructions'), pos_list_metainfo, lemmatizer))

    pos_list_code_comments = []
    code_comments = read_all_comments_from_project(project_folder, 'all_code_comments.csv', pos_list_code_comments,
                                                   lemmatizer)
    all_words.extend(code_comments)

    pos_list_project_comments = []
    project_comments = read_all_comments_from_project(project_folder, 'all_project_comments.csv',
                                                      pos_list_project_comments, lemmatizer)
    all_words.extend(project_comments)

    with open(os.path.join(project_folder, 'pos.json'), 'w') as pos_file:
        dict = {'pos_metainfo': pos_list_metainfo,
                'pos_project_comments': pos_list_project_comments,
                'pos_code_comments': pos_list_code_comments}
        json.dump(dict, pos_file)
    return all_words


def read_all_comments_from_project(project_folder, comment_file, pos_list, lemmatizer):
    comment_strings = []
    comments_file_path = os.path.join(project_folder, comment_file)
    if os.path.isfile(comments_file_path):
        try:
            for idx, row in pd.read_csv(comments_file_path, lineterminator='\n').iterrows():
                comment_strings.extend(preprocess_string(str(row['comment_string']), pos_list, lemmatizer))
        except Exception as e:
            print(e)
    return comment_strings


def preprocess_string(string, pos_list_project, lemmatizer):
    if 'emoji' in string:
        string = __parse_emojis_from_string(string)
    string = urlRegularExpression.sub(' ', string)
    tokenized = TweetTokenizer().tokenize(unescape(string))
    filtered_words = [word.lower() for word in tokenized if
                      word.lower() not in stops and word not in ['`', '\''] and word.isalpha()]
    return [get_lematized_word_and_fill_pos(w, pos_list_project, lemmatizer) for w in filtered_words]


def __parse_emojis_from_string(string):
    emojiRe = re.compile('<img.+ alt="(.*) emoji">')
    emojies = emojiRe.findall(string)
    removed = emojiRe.sub('', string)
    return removed + ' ' + ''.join(emojies)


def get_lematized_word_and_fill_pos(word, pos_list_project, lemmatizer):
    """Map POS tag to first character lemmatize() accepts"""
    pos_tag = nltk.pos_tag([word])[0][1]
    first_letter_of_pos_tag = pos_tag[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    wordnet_tag = tag_dict.get(first_letter_of_pos_tag, wordnet.NOUN)
    lemmatized_word = lemmatizer.lemmatize(word, wordnet_tag)
    pos_list_project.append((lemmatized_word, pos_tag))
    return lemmatized_word


def sentiment_analyser_project_comments(projects):
    csv_file = open('../out/sentiment_analysis.csv', mode='w', encoding='utf8')
    fieldnames = ['project_id', 'number_of_comments', 'number_of_en_comments', 'summary', 'sentiment_mean',
                  'sentiment_complete']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    analyser = SentimentIntensityAnalyzer()
    overall_summary_counter = collections.Counter()
    comment_numbers = []
    en_comment_numbers = []
    for project in projects:
        project_comments = get_all_comments_for_project(project.id)
        if len(project_comments) > 0:
            sentiment_result = __get_sentiment_result(project.id, project_comments, analyser)
            writer.writerow(sentiment_result)
            overall_summary_counter.update(sentiment_result['summary'])
            comment_numbers.append(len(project_comments))
            if sentiment_result['number_of_en_comments'] > 0:
                en_comment_numbers.append(sentiment_result['number_of_en_comments'])

    summary = {'number_of_projects': len(projects),
               'number_of_projects_with_comments': len(comment_numbers),
               'number_of_comments': sum(comment_numbers),
               'number_projects_with_en_comments': len(en_comment_numbers),
               'number_of_en_comments': sum(en_comment_numbers),
               'min_number_of_comments': min(comment_numbers),
               'max_number_of_comments': max(comment_numbers),
               'avg_number_of_comments': sum(comment_numbers) / len(comment_numbers),
               'overall_summary': dict(overall_summary_counter)}
    out_file_summary = open('../out/sentiment_analysis_summary.json', mode='w')
    json.dump(summary, out_file_summary)
    csv_file.close()
    out_file_summary.close()
    return summary


def __get_sentiment_result(project_id, comments, analyser):
    score_counter = collections.Counter()
    summary = {'negative': 0, 'neutral': 0, 'positive': 0}
    all_sentiments = []
    number_of_en_comments = 0
    for comment in comments:
        score = analyser.polarity_scores(comment)
        score_counter.update(score)
        __add_score_to_summary(score, summary)
        lang = get_lang(comment)
        if lang == 'en':
            number_of_en_comments += 1
        all_sentiments.append({'comment': comment, 'score': score, 'lang': lang})
    score_mean = __make_score_count_mean(score_counter, len(comments))
    return {'project_id': project_id,
            'number_of_comments': len(comments),
            'number_of_en_comments': number_of_en_comments,
            'summary': summary,
            'sentiment_mean': score_mean,
            'sentiment_complete': all_sentiments}


def __add_score_to_summary(score, summary):
    if score['compound'] >= 0.05:
        summary['positive'] += 1
    elif score['compound'] <= -0.05:
        summary['negative'] += 1
    else:
        summary['neutral'] += 1


def __make_score_count_mean(score_counter, number_of_scores):
    score_mean = dict(score_counter)
    for key in score_mean.keys():
        score_mean[key] = score_mean[key] / number_of_scores
        return score_mean


def get_lang(string):
    try:
        return detect(string)
    except:
        pass
    return ''


def __get_project_folders(projects_out_path):
    return [folder for folder in os.listdir(projects_out_path) if
            os.path.isdir(os.path.join(projects_out_path, folder))]
