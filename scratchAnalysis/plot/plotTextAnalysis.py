import os

import matplotlib.pyplot as plt
from html import unescape
import json
import math

import nltk
import seaborn as sns
from nltk.corpus import stopwords
from numpy.ma import arange

from scratchAnalysis.analysis.textAnalysis import OUT_FOLDER
from scratchAnalysis.plot.plotSummary import purple_color_3

stops = stopwords.words('english')
sns.set(style='whitegrid', palette=sns.cubehelix_palette(8, reverse=True))


def plot_summary(number_of_neg, number_of_neu, number_of_pos):
    # sentiment_analysis_summary = open('../out/sentiment_analysis_summary.json')
    # sentiment_summary = json.load(sentiment_analysis_summary)
    # counts = sentiment_summary.get('overall_summary')
    plt.bar(['Negativ', 'Neutral', 'Positv'], [number_of_neg, number_of_neu, number_of_pos],
            color=sns.cubehelix_palette(8, reverse=True))
    plt.xlabel("Sentiment")
    plt.ylabel("Anzahl")
    plt.title("Anzahl an Projektkommentaren pro Sentiment")
    plt.tight_layout()
    plt.savefig('../out/summary_plot.pdf')
    plt.savefig('../out/summary_plot.png')
    plt.show()


def plot_project_sentiment(project_out_folder,
                           comment_type):  # plot project according their comment frequency and sentiment
    pos_value = []
    neg_value = []
    s = []
    if 'projects_sentiment' + comment_type + '.json' in os.listdir(OUT_FOLDER):
        with open(os.path.join(OUT_FOLDER, 'projects_sentiment' + comment_type + '.json')) as out_file:
            sentiment_dict = json.load(out_file)
            pos_value = sentiment_dict['pos_value']
            neg_value = sentiment_dict['neg_value']
            s = sentiment_dict['s']
    else:
        for idx, project in enumerate(os.listdir(project_out_folder)):
            pos_score, neg_score, number_comments = __get_sentiment_for_project(
                os.path.join(project_out_folder, project),
                comment_type)
            if number_comments > 0:
                pos_value.append(pos_score)
                neg_value.append(neg_score)
                s.append(number_comments)
            if idx % 10000 == 0:
                print("Done " + str(idx))
        with open(os.path.join(OUT_FOLDER, 'projects_sentiment' + comment_type + '.json'), 'w') as out_file:
            json.dump({'pos_value': pos_value, 'neg_value': neg_value, 's': s}, out_file)
    # u, c = np.unique(np.c_[pos_value, neg_value], return_counts=True, axis=0)
    # print(u)
    # s = lambda x: (((x - x.min()) / float(x.max() - x.min()) + 1) * 8) ** 2

    plt.xlabel("Positiv")
    plt.ylabel("Negativ")
    comment_type_label = "Code"
    if comment_type == 'project_comments':
        comment_type_label = "Projekt"
    plt.title("Projekte in Relation zu ihrer " + comment_type_label + "kommentaranzahl und deren Sentiments")
    plt.scatter(pos_value, neg_value, s=s, color=purple_color_3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig('../out/project_sentiment' + comment_type + '.pdf')
    plt.savefig('../out/project_sentiment' + comment_type + '.png')
    plt.show()
    plt.clf()


def __get_sentiment_for_project(project_folder, comment_type):
    with open(os.path.join(project_folder, comment_type + '_summary.json')) as comment_summary_file:
        comment_summary = json.load(comment_summary_file)
        return comment_summary.get('pos_score_mean'), comment_summary.get('neg_score_mean'), comment_summary.get(
            'number_of_comments')


def __plt_word_frequency(freq_dist, title, out_file_name):
    print([x for x, _ in freq_dist.most_common(20)])
    y_val = [y for _, y in freq_dist.most_common(20)]
    plt.plot(y_val, color=purple_color_3)
    plt.xlabel("Wörter")
    plt.ylabel("Frequenz")
    plt.title(title)
    plt.xticks(arange(20), [x for x, _ in freq_dist.most_common(20)], rotation=45, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig('../out/{0}.pdf'.format(out_file_name))
    plt.savefig('../out/{0}.png'.format(out_file_name))
    #plt.show()
    plt.clf()

    # Zipfs law with log-log plot
    print(title)
    print(y_val)
    print([math.log(i) for i in y_val])
    y_final = []
    for i, k, z, t in zip(y_val[0::4], y_val[1::4], y_val[2::4], y_val[3::4]):
        y_final.append(math.log(i + k + z + t))
    x_val = [math.log(i + 1) for i in range(len(y_final))]
    fig = plt.figure(figsize=(10, 5))
    plt.xlabel("Wörter")
    plt.ylabel("Frequenz (Log)")
    plt.title(title)
    plt.plot([math.log(i) for i in y_val], color=purple_color_3)
    plt.xticks(arange(20), [x for x, _ in freq_dist.most_common(20)], rotation=45, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig('../out/{0}_log.pdf'.format(out_file_name))
    plt.savefig('../out/{0}_log.png'.format(out_file_name))
    plt.show()
    plt.clf()


#  lowercasing, tokenizing, stopword removal for plotting
def __tokenize_text(text):
    tokenized = nltk.tokenize.word_tokenize(unescape(text))
    if 'https' in tokenized:
        print(text)
    return [word.lower() for word in tokenized if word.lower() not in stops and word.isalpha()]


def plt_frequency():
    cumulative = _get_kwarg(kwargs, 'cumulative', False)
    percents = _get_kwarg(kwargs, 'percents', False)
    conditions = [c for c in _get_kwarg(kwargs, 'conditions', self.conditions()) if
                  c in self]  # conditions should be in self
    title = _get_kwarg(kwargs, 'title', '')
    samples = _get_kwarg(
        kwargs, 'samples', sorted(set(v
                                      for c in conditions
                                      for v in self[c]))
    )  # this computation could be wasted
    if "linewidth" not in kwargs:
        kwargs["linewidth"] = 2
    ax = plt.gca()
    if (len(conditions) != 0):
        freqs = []
        for condition in conditions:
            if cumulative:
                # freqs should be a list of list where each sub list will be a frequency of a condition
                freqs.append(list(self[condition]._cumulative_frequencies(samples)))
                ylabel = "Cumulative Counts"
                legend_loc = 'lower right'
                if percents:
                    freqs[-1] = [f / freqs[len(freqs) - 1] * 100 for f in freqs]
                    ylabel = "Cumulative Percents"
            else:
                freqs.append([self[condition][sample] for sample in samples])
                ylabel = "Counts"
                legend_loc = 'upper right'
            # percents = [f * 100 for f in freqs] only in ConditionalProbDist?

        i = 0
        for freq in freqs:
            kwargs['label'] = conditions[i]  # label for each condition
            i += 1
            ax.plot(freq, *args, **kwargs)
        ax.legend(loc=legend_loc)
        ax.grid(True, color="silver")
        ax.set_xticks(range(len(samples)))
        ax.set_xticklabels([str(s) for s in samples], rotation=90)
        if title:
            ax.set_title(title)
        ax.set_xlabel("Samples")
        ax.set_ylabel(ylabel)
    plt.show()