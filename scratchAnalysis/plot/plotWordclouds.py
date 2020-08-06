import ast
import csv
import json
import os

import nltk

from scratchAnalysis.analysis.textAnalysis import get_tfidf_vectorizer, get_list_of_words, \
    read_all_strings_from_project, read_all_strings_from_projects, read_all_comments_from_projects, preprocess_string, \
    read_description_and_title_from_projects
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scratchAnalysis.plot.plotTextAnalysis import __plt_word_frequency, __tokenize_text, plot_summary

OUT_FOLDER = "C:/Users/Isabella/Documents/Masterthesis/Code/out/wordclouds"


def generate_tfidf_word_cloud(projects_out_path):
    vectorizer, vector, documents = get_tfidf_vectorizer(projects_out_path)
    feature_names = vectorizer.get_feature_names()

    wordcloud_dict = dict()
    for doc_number in range(len(documents)):
        feature_index = vector[doc_number, :].nonzero()[1]
        tfidf_scores = zip(feature_index, [vector[doc_number, x] for x in feature_index])
        for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
            if w not in wordcloud_dict or s > wordcloud_dict[w]:
                wordcloud_dict[w] = s
        if doc_number % 10000 == 0:
            print('Tfidf_wordcloud done: ' + str(doc_number) + ' of ' + str(len(documents)))
    #df_idf = pd.DataFrame(vector, index=feature_names, columns=["tfidf_weights"])
    #print(df_idf.sort_values(by='tfidf_weights', ascending=False).head(30))

    print(wordcloud_dict)

    wordcloud = WordCloud(width=1000, height=500, background_color='white', scale=2,
                          collocations=False).generate_from_frequencies(wordcloud_dict)
    wordcloud.to_file(os.path.join(OUT_FOLDER, "wordcloud_tfidf.pdf"))



def display_scores(vectorizer, tfidf_result):
    # http://stackoverflow.com/questions/16078015/
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    print("display_scores")
    for idx, item in enumerate(sorted_scores):
        print("{0:50} Score: {1}".format(item[0], item[1]))
        if idx > 100:
            return


def green_color_wordcloud(word, font_size, position, orientation, random_state=None,
                          **kwargs):
    return "hsl(100%%,100%%, %d%%" % np.random.randint(1, 51)


def generate_wordcloud_for_code_comments(projects_out_path, out_folder):
    text = read_all_comments_from_projects(projects_out_path, 'all_code_comments.csv')
    out_file = os.path.join(out_folder, "wordclouds/wordcloudCodeComments")
    print(out_file)
    generate_wordcloud_for_text(text, out_file)


def generate_wordcloud_for_project_comments(projects_out_path, out_folder):
    text = read_all_comments_from_projects(projects_out_path, 'all_project_comments.csv')
    out_file = os.path.join(out_folder, "wordclouds/wordcloudProjectComments")
    print(out_file)
    generate_wordcloud_for_text(text, out_file)


def generate_wordcloud_for_projects(projects_out_path, out_folder):  # visualizing most popular words
    text = read_all_strings_from_projects(projects_out_path)
    print(out_folder)
    out_file = os.path.join(out_folder, "wordclouds/wordcloudAllText")
    print(out_file)
    generate_wordcloud_for_text(text, out_file)


def generate_metainfo_wordcloud_for_projects(projects_out_path, out_folder):  # visualizing most popular words
    text = read_description_and_title_from_projects(projects_out_path)

    print(out_folder)
    out_file = os.path.join(out_folder, "wordclouds/wordcloudAllMetainfoText")
    print(out_file)
    generate_wordcloud_for_text(text, out_file)


def generate_wordcloud_for_text(text, out_file):
    wordcloud = generate_wordcloud(text)
    wordcloud.to_file(out_file + '.pdf')
    wordcloud.to_file(out_file + '.png')


# TODO nur project Kommentare? - Ja Sentiment ergibt sonst keinen sinn
def plot_wordclouds(projects_folder):
    negative_tokens = []
    positive_tokens = []
    neutral_tokens = []
    number_of_neg_comments = 0
    number_of_pos_comments = 0
    number_of_neu_comments = 0
    number_of_projects_with_comments = 0
    number_of_projects = len(os.listdir(projects_folder))
    print('Plot wordclouds of: ' + str(projects_folder))
    if 'out_plt_wordclouds.json' in os.listdir(OUT_FOLDER):
        with open(os.path.join(OUT_FOLDER, 'out_plt_wordclouds.json')) as out_plt_wordclouds_file:
            out_words = json.load(out_plt_wordclouds_file)
            positive_tokens = out_words['positive_tokens']
            negative_tokens = out_words['negative_tokens']
            neutral_tokens = out_words['neutral_tokens']
            number_of_neg_comments = out_words['number_of_neg_comments']
            number_of_pos_comments = out_words['number_of_pos_comments']
            number_of_neu_comments = out_words['number_of_neu_comments']
    else:
        for idx, project_folder in enumerate(os.listdir(projects_folder)):
            project_path = os.path.join(projects_folder, project_folder)
            if idx % 10000 == 0:
                print("Done : " + str(idx) + ' of ' + str(number_of_projects))
            if "all_project_comments.csv" in os.listdir(project_path):
                number_of_projects_with_comments += 1
                all_project_comments_df = pd.read_csv(os.path.join(project_path, "all_project_comments.csv"))
                for index, row in all_project_comments_df.iterrows():
                    type = __get_sentiment_type_from_compound_score(row['compound'])
                    processed_text = preprocess_string(str(row['comment_string']), [], WordNetLemmatizer())
                    if type == 'neg':
                        negative_tokens.extend(processed_text)
                        number_of_neg_comments += 1
                    elif type == 'pos':
                        positive_tokens.extend(processed_text)
                        number_of_pos_comments += 1
                    else:
                        neutral_tokens.extend(processed_text)
                        number_of_neu_comments += 1
        print("Number of projects with project comments: " + str(number_of_projects_with_comments))
    print("Länge Neg: " + str(len(negative_tokens)))
    print("Länge Pos: " + str(len(positive_tokens)))
    print("Länge Neu: " + str(len(neutral_tokens)))
    print("Länge Neg unique: " + str(len(set(negative_tokens))))
    print("Länge Pos unique: " + str(len(set(positive_tokens))))
    print("Länge Neu unique: " + str(len(set(neutral_tokens))))
    if 'out_plt_wordclouds.json' not in os.listdir(OUT_FOLDER):
        with open(os.path.join(OUT_FOLDER, 'out_plt_wordclouds.json'), 'w') as out_plt_wordclouds_file:
            print('write out')
            out_words = {'negative_tokens': negative_tokens, 'positive_tokens': positive_tokens,
                         'neutral_tokens': neutral_tokens,
                         'number_of_neg_comments': number_of_neg_comments,
                         'number_of_pos_comments': number_of_pos_comments,
                         'number_of_neu_comments': number_of_neu_comments}
            json.dump(out_words, out_plt_wordclouds_file)
    plot_summary(number_of_neg_comments, number_of_neu_comments, number_of_pos_comments)
    negative_freq = nltk.FreqDist(negative_tokens)
    print(negative_freq)
    positive_freq = nltk.FreqDist(positive_tokens)
    neutral_freq = nltk.FreqDist(neutral_tokens)
    __plt_word_frequency(negative_freq, 'Wort Frequenz Verteilung (Negativ)', 'sentiment_neg_frequency')
    __plt_word_frequency(positive_freq, 'Wort Frequenz Verteilung (Positiv)', 'sentiment_pos_frequency')
    __plt_word_frequency(neutral_freq, 'Wort Frequenz Verteilung (Neutral)', 'sentiment_neu_frequency')
    wordcloud_neg = generate_wordcloud_with_colorfunc(negative_tokens, red_color_func)
    wordcloud_neg.to_file(os.path.join(OUT_FOLDER, "wordcloudNegProjectComments.png"))
    wordcloud_neg.to_file(os.path.join(OUT_FOLDER, "wordcloudNegProjectComments.pdf"))
    wordcloud_pos = generate_wordcloud_with_colorfunc(positive_tokens, green_color_func)
    wordcloud_pos.to_file(os.path.join(OUT_FOLDER, "wordcloudPosProjectComments.png"))
    wordcloud_pos.to_file(os.path.join(OUT_FOLDER, "wordcloudPosProjectComments.pdf"))
    wordcloud_neu = generate_wordcloud_with_colorfunc(neutral_tokens, grey_color_func)
    wordcloud_neu.to_file(os.path.join(OUT_FOLDER, "wordcloudNeuProjectComments.png"))
    wordcloud_neu.to_file(os.path.join(OUT_FOLDER, "wordcloudNeuProjectComments.pdf"))


def generate_wordcloud(text):
    wordcloud = WordCloud(width=1000, height=500, background_color='white', scale=2, collocations=False).generate(
        " ".join(text))  # liste an wörter übergeben
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    return wordcloud


def generate_wordcloud_with_colorfunc(text, colorfunc):
    wordcloud = WordCloud(width=1000, height=500, background_color='white', scale=2, color_func=colorfunc,
                          collocations=False).generate(
        " ".join(text))  # liste an wörter übergeben
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    return wordcloud


def __get_sentiment_type_from_compound_score(compound_score):
    if compound_score >= 0.05:
        return 'pos'
    elif compound_score <= -0.05:
        return 'neg'
    else:
        return 'neu'


def red_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
    h = int(6)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 110)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)


def green_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
    h = int(144)
    s = int(100)
    l = int(100.0 * float(random_state.randint(30, 80)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)


def grey_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
    h = int(0)
    s = int(0)
    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)
