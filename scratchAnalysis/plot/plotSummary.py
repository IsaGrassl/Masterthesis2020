import json
import os
from collections import Counter

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import datetime

from scratchAnalysis.analysis.textAnalysis import get_lang

sns.set(style='whitegrid', palette=sns.cubehelix_palette(8, reverse=True))

purple_color_2 = [0.3210194743259347, 0.19303051265196464, 0.3707881677724792]
purple_color_3 = [0.46496993672552045, 0.26868986121314253, 0.4636527763640647]
purple_color_4 = [0.6046906802634469, 0.35739308184976665, 0.5337407853692406]
purple_color_5 = [0.7195800708349119, 0.45537982893127477, 0.5861062995810926]


def plot_read_all_summary(out_path):
    with open(os.path.join(out_path, 'summary.json')) as summary_file:
        summary = json.load(summary_file)
        print("Number of projects: " + str(len(summary['project_creation_dates'])))
        print("Number of projects with code comments: " + str(
            len([number for number in summary['code_comment_word_counts'] if number != 0])))
        print("Number of projects with project comments: " + str(
            len([number for number in summary['project_comment_word_counts'] if number != 0])))

        plot_creation_dates(summary['project_creation_dates'], out_path)
        plot_opcode_numbers(summary['opcode_counts'], out_path)
        plot_word_numbers(summary['code_comment_word_counts'], out_path, 'Anzahl der Wörter',
                          'Verteilung der Wörter in Codekommentaren pro Projekt', 'distribution_code_comment_words')
        plot_word_numbers(summary['project_comment_word_counts'], out_path, 'Anzahl der Wörter',
                          'Verteilung der Wörter in Projektkommentaren pro Projekt',
                          'distribution_project_comment_words')


def plot_creation_dates(creation_dates, out_path):
    plot_dates(creation_dates, os.path.join(out_path, 'distribution_projects_creation_time'),
               'Verteilung der Projekte über Zeitraum')


def plot_dates(dates, out_file, title):
    formatted_dates = [datetime.datetime.fromisoformat(creation_date[:-1]).date() for creation_date in dates]
    print(formatted_dates)
    counter = Counter(formatted_dates)
    data = {'dates': list(counter.keys()),
            'counts': list(counter.values())}
    ax = sns.lineplot(x='dates', y='counts', data=data, color=purple_color_3)
    ax.set(xlabel='Datum', ylabel='Anzahl der Projekte')
    ax.set_title(title)
    plt.xlim(datetime.date(2020, 2, 1), datetime.date(2020, 5, 1))
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig(out_file + '.pdf')
    plt.savefig(out_file + '.png')
    plt.show()
    plt.clf()


def plot_word_numbers(word_numbers, out_path, x_label, title, out_name):
    ax = sns.distplot(word_numbers, kde=False, hist_kws={'log': True, 'alpha': 1}, color=purple_color_3)

    ax.set(xlabel=x_label, ylabel='Anzahl der Projekte (log)')
    ax.set_title(title)
    plt.savefig(os.path.join(out_path, out_name + '.pdf'))
    plt.savefig(os.path.join(out_path, out_name + '.png'))
    plt.show()
    plt.clf()


def plot_opcode_numbers(opcodes_numbers, out_path):
    ax = sns.distplot(opcodes_numbers, kde=False, hist_kws={'log': True, 'alpha': 1}, color=purple_color_3)

    ax.set(xlabel='Anzahl der Opcodes', ylabel='Anzahl der Projekte (log)')
    ax.set_title('Verteilung der Opcodes')
    plt.savefig(os.path.join(out_path, 'distribution_number_opcodes.pdf'))
    plt.savefig(os.path.join(out_path, 'distribution_number_opcodes.png'))
    plt.show()
    plt.clf()


def read_pos(projects_path, out_path):
    metainfo_pos_counter = Counter()
    code_comment_pos_counter = Counter()
    project_comment_pos_counter = Counter()
    pos_counter_path = os.path.join(out_path, 'pos_counter.json')
    if os.path.isfile(pos_counter_path):
        with open(pos_counter_path) as pos_counter_file:
            return json.load(pos_counter_file)
    for idx, project_folder in enumerate(os.listdir(projects_path)):
        project_path = os.path.join(projects_path, project_folder)
        if idx % 10000 == 0:
            print("pos_counter: " + str(idx))
        if 'pos.json' in os.listdir(project_path):
            with open(os.path.join(project_path, 'pos.json')) as pos_file:
                pos_info = json.load(pos_file)
                metainfo_pos_counter = metainfo_pos_counter + Counter([pos for _, pos in pos_info['pos_metainfo']])
                code_comment_pos_counter = code_comment_pos_counter + Counter(
                    [pos for _, pos in pos_info['pos_code_comments']])
                project_comment_pos_counter = project_comment_pos_counter + Counter(
                    [pos for _, pos in pos_info['pos_project_comments']])

        else:
            print('no pos')
    pos_counter = {'metainfo_pos_counter': metainfo_pos_counter,
                   'code_comment_pos_counter': code_comment_pos_counter,
                   'project_comment_pos_counter': project_comment_pos_counter}
    with open(pos_counter_path, 'w') as pos_counter_file:
        json.dump(pos_counter, pos_counter_file)
    return pos_counter


def make_summary(projects_path, out_path):
    summary = {
        "project_creation_dates": [],
        "opcode_counts": [],
        "metainfo_title_word_counts": [],
        "metainfo_instructions_word_counts": [],
        "metainfo_description_word_counts": [],
        "metainfo_number_of_english_titles": 0,
        "metainfo_number_of_english_instructions": 0,
        "metainfo_number_of_english_description": 0,
        "metainfo_number_of_english": 0,
        "code_comment_word_counts": [],
        "code_comment_total_counts": [],
        "code_comment_english_counts": [],
        "project_comment_word_counts": [],
        "project_comment_total_counts": [],
        "project_comment_english_counts": [],
        "number_of_projects_with_english_project_comments": 0,
        "number_of_projects_with_english_code_comments": 0
    }
    summary_path = os.path.join(out_path, 'summary.json')
    if os.path.isfile(summary_path):
        with open(summary_path) as summary_file:
            return json.load(summary_file)
    else:
        for idx, project_folder in enumerate(os.listdir(projects_path)):
            project_path = os.path.join(projects_path, project_folder)
            project_path_list = os.listdir(project_path)
            if 'project_metainfo.json' in project_path_list:
                with open(os.path.join(project_path, 'project_metainfo.json')) as metainfo_file:
                    metainfo = json.load(metainfo_file)
                    summary["project_creation_dates"].append(metainfo.get('history').get('shared'))
                    metainfo_title = metainfo.get('title', '')
                    summary['metainfo_title_word_counts'].append(len(str(metainfo_title).split()))
                    is_english_title = get_is_english_string(metainfo_title)
                    summary['metainfo_number_of_english_titles'] += is_english_title
                    metainfo_instructions = metainfo.get('instructions', '')
                    summary['metainfo_instructions_word_counts'].append(len(str(metainfo_instructions).split()))
                    is_english_instructions = get_is_english_string(metainfo_instructions)
                    summary['metainfo_number_of_english_instructions'] += is_english_instructions
                    metainfo_description = metainfo.get('description', '')
                    summary['metainfo_description_word_counts'].append(len(str(metainfo_description).split()))
                    is_english_description = get_is_english_string(metainfo_description)
                    summary['metainfo_number_of_english_description'] += is_english_description
                    if is_english_title + is_english_instructions + is_english_description > 0:
                        summary['metainfo_number_of_english'] += 1
            if 'all_project_comments.csv' in project_path_list:
                number_of_words, number_total, has_english, number_of_english = get_numbers_of_comments(
                    os.path.join(project_path, 'all_project_comments.csv'))
                summary['project_comment_word_counts'].append(number_of_words)
                summary['project_comment_total_counts'].append(number_total)
                summary['project_comment_english_counts'].append(number_of_english)
                if has_english:
                    summary['number_of_projects_with_english_project_comments'] += 1
            if 'all_code_comments.csv' in project_path_list:
                number_of_words, number_total, has_english, number_of_english = get_numbers_of_comments(
                    os.path.join(project_path, 'all_code_comments.csv'))
                summary['code_comment_word_counts'].append(number_of_words)
                summary['code_comment_total_counts'].append(number_total)
                summary['code_comment_english_counts'].append(number_of_english)
                if has_english:
                    summary['number_of_projects_with_english_code_comments'] += 1
            if idx % 10000 == 0:
                print("make_summary: " + str(idx))
        opcodes_df = pd.read_csv(os.path.join(out_path, 'project_opcodes.csv'))
        transposed = opcodes_df.drop('project_id', 1).transpose()
        summary['opcode_counts'] = transposed.sum().tolist()

        with open(summary_path, 'w') as summary_file:
            json.dump(summary, summary_file)
        return summary


def get_numbers_of_comments(comments_file_path):
    number_of_words = 0
    number_of_english = 0
    number_total = 0
    has_english = False
    if os.path.isfile(comments_file_path):
        try:
            for idx, row in pd.read_csv(comments_file_path, lineterminator='\n').iterrows():
                number_of_words += len(str(row['comment_string']).split())
                if row['is_english'] is True:
                    number_of_english += 1
                    has_english = True
                number_total += 1
        except Exception as e:
            print(e)
    return number_of_words, number_total, has_english, number_of_english


def get_is_english_string(string):
    lang = get_lang(string)
    return 1 if lang == 'en' else 0
