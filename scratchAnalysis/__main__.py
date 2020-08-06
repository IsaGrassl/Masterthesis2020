import json
import os
from collections import Counter
import pandas as pd
import threading

from scratchAnalysis.analysis.kMeans import load_opcodes_df, clustering_opcodes, elbow_opcodes, correlationmatrix, \
    make_opcodes_general_df

# from readProjects.scratch import ScratchProject
from scratchAnalysis.analysis.keywordAnalysis import get_projects_with_keywords, get_all_project_creation_dates
from scratchAnalysis.analysis.textAnalysis import get_tfidf_features
from scratchAnalysis.plot.plotOpcodes import plot_general_opcodes_for_df_with_labels, plot_opcodes_for_df_with_labels, \
    plt_most_common_opcodes, plt_boxplot_general_opcodes, plt_all_general_opcodes_df
from scratchAnalysis.plot.plotSummary import plot_dates, make_summary
from scratchAnalysis.plot.plotTextAnalysis import plot_project_sentiment
from scratchAnalysis.plot.plotWordclouds import generate_tfidf_word_cloud, generate_wordcloud_for_projects, \
    generate_wordcloud_for_code_comments, generate_wordcloud_for_project_comments, plot_wordclouds, \
    generate_metainfo_wordcloud_for_projects


def read_all_projects(files):
    projects = []
    for file in files:
        try:
            # project = ScratchProject(os.path.join(path, file))
            # projects.append(project)
            print('waddup')
        except:
            print("ERROR: Folgendes Projekt konnte nicht eingelesen werden: " + os.path.join(path, file))
    return projects


def get_all_opcode_counters(opcodes_df):
    print(opcodes_df.sum())
    all_opcodes_counter = Counter()
    for idx, row in opcodes_df.iterrows():
        all_opcodes_counter += Counter(row)
    return all_opcodes_counter


def get_general_opcode_counters(general_opcodes_df):
    all_opcodes_general_counter = Counter()
    all_opcodes_general = dict()
    for idx, row in general_opcodes_df.iterrows():
        count_opcodes_general = Counter(row)
        all_opcodes_general_counter += count_opcodes_general
        for key in count_opcodes_general:
            if key not in all_opcodes_general.keys():
                all_opcodes_general[key] = []
            all_opcodes_general[key].append(count_opcodes_general[key])
    return all_opcodes_general_counter, all_opcodes_general


def get_all_vectors(projects):
    vectors = []
    for project in projects:
        vectors.append(project.generate_vector())
    return vectors


def reduce_opcodes(opcodes_df):
    opcodes_dict = dict()
    for opcode in os.listdir(projects_all_out_path):
        opcodes_dict[opcode] = 1
    df = opcodes_df.drop(opcodes_df[opcodes_df['project_id'].apply(lambda x: opcodes_dict.get(str(x), 0)) == 0].index)
    df.to_csv(os.path.join(out_path, 'project_opcodesV2.csv'))


def clustering_helper(opcodes_df, prefix, norm):
    opcodes_df_with_labels = clustering_opcodes(opcodes_df, prefix, norm)
    opcodes_df_with_labels.to_csv(os.path.join(out_path, 'labeled_opcodes' + prefix + '.csv'))
    opcodes_df_with_labels[['project_id', 'label']].to_csv(
        os.path.join(out_path, 'clustering_project_opcodes' + prefix + '.csv'))
    return opcodes_df_with_labels


def clustering():
    opcodes_df = load_opcodes_df(out_path).drop('Unnamed: 0', 1)
    opcodes_general_df = make_opcodes_general_df(opcodes_df)
    pd.DataFrame.from_dict({'sum_opcodes': opcodes_df.sum()}).to_csv(os.path.join(out_path, 'test_sum.csv'))
    return
    opcodes_df['sum_opcodes'] = opcodes_df.drop('project_id', 1).transpose().sum()
    opcodes_df_75 = opcodes_df[opcodes_df['sum_opcodes'] < opcodes_df['sum_opcodes'].quantile(.75)].drop('sum_opcodes',
                                                                                                         1)
    opcodes_df_25 = opcodes_df[opcodes_df['sum_opcodes'] > opcodes_df['sum_opcodes'].quantile(.25)].drop('sum_opcodes',
                                                                                                         1)
    # reduce_opcodes(opcodes_df)
    print(opcodes_df)
    opcodes_general_df_75 = make_opcodes_general_df(opcodes_df)

    #print('l1_75')
    #opcodes_df_with_labels = clustering_helper(opcodes_df_75, 'l1_75', 'l1')
    opcodes_df_with_labels = pd.read_csv(os.path.join(out_path, 'labeled_opcodesl1_75.csv')).drop('Unnamed: 0', 1)
    plot_opcodes_for_df_with_labels(opcodes_df_with_labels, 'l1_75')
    #opcodes_df_with_labels = clustering_helper(opcodes_df_75, 'l2_75', 'l2')
    opcodes_df_with_labels = pd.read_csv(os.path.join(out_path, 'labeled_opcodesl2_75.csv')).drop('Unnamed: 0', 1)
    plot_opcodes_for_df_with_labels(opcodes_df_with_labels, 'l2_75')

    #print('l1_75_general')
    #opcodes_df_with_labels = clustering_helper(opcodes_general_df_75, 'l1_75_general', 'l1')
    opcodes_df_with_labels = pd.read_csv(os.path.join(out_path, 'labeled_opcodesl1_75_general.csv')).drop('Unnamed: 0', 1)
    plot_general_opcodes_for_df_with_labels(opcodes_df_with_labels, 'l1_75_general')
    #opcodes_df_with_labels = clustering_helper(opcodes_general_df_75, 'l2_75_general', 'l2')
    opcodes_df_with_labels = pd.read_csv(os.path.join(out_path, 'labeled_opcodesl2_75_general.csv')).drop('Unnamed: 0', 1)
    plot_general_opcodes_for_df_with_labels(opcodes_df_with_labels, 'l2_75_general')


    #print('l1_25')
    #opcodes_df_with_labels = clustering_helper(opcodes_df_25, 'l1_25', 'l1')
    #opcodes_df_with_labels = pd.read_csv(os.path.join(out_path, 'labeled_opcodesl1_75.csv')).drop('Unnamed: 0', 1)
    #plot_opcodes_for_df_with_labels(opcodes_df_with_labels, 'l1_25')
    #opcodes_df_with_labels = clustering_helper(opcodes_df_25, 'l2_25', 'l2')
    #opcodes_df_with_labels = pd.read_csv(os.path.join(out_path, 'labeled_opcodesl2_75.csv')).drop('Unnamed: 0', 1)
    #plot_opcodes_for_df_with_labels(opcodes_df_with_labels, 'l2_25')

    print('l1_25_general')
    opcodes_df_with_labels = clustering_helper(opcodes_general_df_75, 'l1_25_general', 'l1')
    #opcodes_df_with_labels = pd.read_csv(os.path.join(out_path, 'labeled_opcodesl1_75_general.csv')).drop('Unnamed: 0', 1)
    plot_general_opcodes_for_df_with_labels(opcodes_df_with_labels, 'l1_25_general')
    opcodes_df_with_labels = clustering_helper(opcodes_general_df_75, 'l2_25_general', 'l2')
    #opcodes_df_with_labels = pd.read_csv(os.path.join(out_path, 'labeled_opcodesl2_75_general.csv')).drop('Unnamed: 0', 1)
    plot_general_opcodes_for_df_with_labels(opcodes_df_with_labels, 'l2_25_general')


    opcodes_df_25_75 = opcodes_df[(opcodes_df['sum_opcodes'] > opcodes_df['sum_opcodes'].quantile(.25)) & (
                opcodes_df['sum_opcodes'] < opcodes_df['sum_opcodes'].quantile(.75))].drop('sum_opcodes', 1)
    opcodes_general_df_25_75 = make_opcodes_general_df(opcodes_df_25_75)

    #print('l1_25_75')
    #opcodes_df_with_labels = clustering_helper(opcodes_df_25_75, 'l1_25_75', 'l1')
    #opcodes_df_with_labels = pd.read_csv(os.path.join(out_path, 'labeled_opcodesl1_25_75.csv')).drop('Unnamed: 0', 1)
    #plot_opcodes_for_df_with_labels(opcodes_df_with_labels, 'l1_25_75')
    #opcodes_df_with_labels = clustering_helper(opcodes_df_25_75, 'l2_25_75', 'l2')
    #opcodes_df_with_labels = pd.read_csv(os.path.join(out_path, 'labeled_opcodesl2_25_75.csv')).drop('Unnamed: 0', 1)
    #plot_opcodes_for_df_with_labels(opcodes_df_with_labels, 'l2_25_75')

    #print('l1_25_75_general')
    #opcodes_df_with_labels = clustering_helper(opcodes_general_df_25_75, 'l1_25_75_general', 'l1')
    #opcodes_df_with_labels = pd.read_csv(os.path.join(out_path, 'labeled_opcodesl1_25_75_general.csv')).drop('Unnamed: 0', 1)
    #plot_general_opcodes_for_df_with_labels(opcodes_df_with_labels, 'l1_25_75_general')
    #opcodes_df_with_labels = clustering_helper(opcodes_general_df_25_75, 'l2_25_75_general', 'l2')
    #opcodes_df_with_labels = pd.read_csv(os.path.join(out_path, 'labeled_opcodesl2_25_75_general.csv')).drop('Unnamed: 0', 1)
    #plot_general_opcodes_for_df_with_labels(opcodes_df_with_labels, 'l2_25_75_general')


    # plt_all_general_opcodes_df(opcodes_general_df)
    # plt_most_common_opcodes(opcodes_df)
    # plt_boxplot_general_opcodes(opcodes_general_df)

    # opcodes_df_with_labels = pd.read_csv(os.path.join(out_path, 'labeled_opcodes.csv'))
    # opcodes_df_with_labels = clustering_opcodes(opcodes_df, opcodes_df_with_labels)
    # opcodes_df_with_labels.to_csv(os.path.join(out_path, 'labeled_opcodes.csv'))
    # opcodes_df_with_labels[['project_id', 'label']].to_csv(os.path.join(out_path, 'clustering_project_opcodes.csv'))

    # plot_opcodes_for_df_with_labels(opcodes_df_with_labels)
    # elbow_opcodes(opcodes_df)
    # correlationmatrix(opcodes_df)

    # opcodes_general_df_with_labels = clustering_opcodes(opcodes_general_df)
    # opcodes_general_df_with_labels = pd.read_csv(os.path.join(out_path, 'labeled_opcodes_general.csv'))
    # opcodes_general_df_with_labels.to_csv(os.path.join(out_path, 'labeled_opcodes_general.csv'))
    # opcodes_general_df_with_labels[['project_id', 'label']].to_csv(os.path.join(out_path, 'clustering_project_opcodes_general.csv'))

    # plot_general_opcodes_for_df_with_labels(opcodes_general_df_with_labels)
    # elbow_opcodes(opcodes_general_df)
    # correlationmatrix(opcodes_general_df)


def reduce_pos_dict_to_counter(dict):
    counter = Counter()
    total_words = 0
    for key in dict.keys():
        total_words += int(dict[key])
        reduced_key = key[:2]
        counter[reduced_key] = counter.get(reduced_key, 0) + int(dict[key])
    print(total_words)
    return counter


if __name__ == "__main__":
    out_path = "C:/Users/Isabella/Documents/Masterthesis/Code/out/"
    out_plots = "C:/Users/Isabella/Documents/Masterthesis/Code/out/plots"
    out = "C:/Users/Isabella/Documents/Masterthesis/Code/out"
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples/sample/randomsample")
    #    files = os.listdir(path)

    projects_out_path = os.path.join('C:\\Users\\Isabella\\Documents\\', 'projects')
    projects_all_out_path = os.path.join('C:\\Users\\Isabella\\Documents\\projectsAll\\', 'projects')
    #clustering()

    # projects_with_keywords = get_projects_with_keywords(['spikeballs'], projects_all_out_path)
    # print(projects_with_keywords)
    # pd.DataFrame.from_dict(projects_with_keywords).to_csv(os.path.join(out_path, 'spikeballs_dates.csv'))
    # covid_dates = pd.read_csv(os.path.join(out_path, 'corona_dates.csv'))
    # plot_dates(covid_dates['creation_dates'], os.path.join(out_plots, 'distribution_covid_1000'), 'Projekte mit Keyword Corona')

    # project_creation_dates = get_all_project_creation_dates(projects_all_out_path)
    # pd.DataFrame.from_dict(project_creation_dates).to_csv(os.path.join(out_path, 'creation_dates.csv'))
    # project_creation_dates = pd.read_csv(os.path.join(out_path, 'creation_dates.csv'))
    # plot_dates(project_creation_dates['creation_dates'], os.path.join(out_plots, 'distribution_all'), 'Projekt Erstellungs Datums')

    # generate_tfidf_word_cloud(projects_all_out_path)
    # generate_wordcloud_for_projects(projects_out_path, out)
    # generate_metainfo_wordcloud_for_projects(projects_all_out_path, out)
    # generate_wordcloud_for_code_comments(projects_out_path, out)
    # generate_wordcloud_for_project_comments(projects_out_path, out)
    # plot_wordclouds(projects_out_path)
    # projects = read_all_projects(files)

    # pos_counter = read_pos(projects_out_path, out_path)
    # pos_counter['metainfo_pos_counter'] = reduce_pos_dict_to_counter(pos_counter['metainfo_pos_counter']).most_common(12)
    # pos_counter['project_comment_pos_counter'] = reduce_pos_dict_to_counter(pos_counter['project_comment_pos_counter']).most_common(12)
    # pos_counter['code_comment_pos_counter'] = reduce_pos_dict_to_counter(pos_counter['code_comment_pos_counter']).most_common(12)
    # print(pos_counter)
    # with open(os.path.join(out_path, 'pos_counter_reduced.json'), 'w') as pos_counter_file:
    #    json.dump(pos_counter, pos_counter_file)

    # print(Counter(pos_counter['metainfo_pos_counter']).most_common(10))
    # print(pos_counter)
    # summary = make_summary(projects_all_out_path, out_path)
    # plot_read_all_summary(out_path)

    # plot_project_sentiment(projects_out_path, 'project_comments')
    # plot_project_sentiment(projects_out_path, 'code_comments')
    get_tfidf_features(projects_all_out_path)
    # analyze_comments(projects)
    # generate_wordcloud_for_projects(projects)
    # sentiment_analyser_project_comments(projects)
    # plot_summary()
    # plot_wordclouds()
