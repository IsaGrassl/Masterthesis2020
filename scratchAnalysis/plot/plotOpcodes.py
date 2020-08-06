import math
import os

import matplotlib.pyplot as plt
import numpy
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

from scratchAnalysis.analysis.kMeans import make_to_general_opcode, remove_uninteresting_general_opcodes, OUT_PLOTS
from scratchAnalysis.plot.plotSummary import purple_color_3, purple_color_4, purple_color_5

sns.set(style='whitegrid', palette=sns.cubehelix_palette(8, reverse=True))


def plt_all_general_opcodes_df(all_opcodes_general_df):
    plt.title('Anzahl der kategorisierten Opcodes')
    df_tmp = all_opcodes_general_df.drop('project_id', 1).sum().reset_index(name='Anzahl der Opcodes').sort_values(
        by='Anzahl der Opcodes', ascending=False)
    print(df_tmp)
    ax = sns.barplot(x="opcodes", y='Anzahl der Opcodes', data=df_tmp)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.xlabel('Opcode')
    # ax.ticklabel_format(axis='y', style='plain')
    plt.tight_layout()
    plt.savefig('../out/count_opcodes_general.pdf')
    plt.savefig('../out/count_opcodes_general.png')
    plt.show()
    plt.clf()


def plt_most_common_opcodes(all_opcodes_df):
    plt.xlabel('Opcode ')
    plt.ylabel('Anzahl der Opcodes')
    plt.title('Anzahl der 20 häufigsten Opcodes')
    ax = all_opcodes_df.drop('project_id', 1).sum().sort_values(ascending=False).head(20).plot.bar(color=purple_color_3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig('../out/count_opcodes.pdf')
    plt.savefig('../out/count_opcodes.png')
    plt.show()
    plt.clf()


# TODO log funktioniert noch nciht ganz
def plt_boxplot_general_opcodes(all_opcodes_general_df):
    # plt.yscale('log')
    df_tmp = all_opcodes_general_df.drop('project_id', 1)
    df_tmp = df_tmp.transpose()
    df_tmp['opcodes'] = df_tmp.index
    df_tmp = df_tmp.melt(id_vars=['opcodes'])

    ax = sns.boxplot(x='opcodes', y='value', data=df_tmp,
                     order=['event', 'control', 'looks', 'motion', 'data', 'sound', 'procedures'])
    ax.set_yscale('symlog')
    ax.set(ylim=(-1, 4000))
    plt.xlabel('Opcode')
    plt.ylabel('Anzahl der Opcodes')
    plt.title('Verteilung der kategorisierten Opcodes')
    plt.tight_layout()
    plt.savefig('../out/distribution_opcodes.pdf')
    plt.savefig('../out/distribution_opcodes.png')
    plt.show()
    plt.clf()


def plot_opcodes_for_df_with_labels(labeled_opcode_df, prefix=''):
    # Nur zur verifikation
    print_number_of_projects_per_label(labeled_opcode_df)

    opcodes_df_without_id = labeled_opcode_df.drop('project_id', 1)
    result_for_labels = opcodes_df_without_id.groupby('label').agg('sum').transpose()
    result_for_labels['total opcode count'] = result_for_labels[0] + result_for_labels[1] + result_for_labels[2]
    result_for_labels['opcodes'] = result_for_labels.index

    result_top20 = result_for_labels.sort_values('total opcode count', ascending=False).head(20)
    result_for_plot = result_top20.melt(id_vars=['opcodes'], value_vars=[0, 1, 2]).rename(columns={
        'opcodes': 'Opcodes', 'label': 'Cluster', 'value': 'Anzahl',
    })
    ax = sns.barplot(x="Opcodes", y="Anzahl", hue="Cluster", data=result_for_plot, palette=[purple_color_3, purple_color_4, purple_color_5])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Anzahl der 20 häufigsten Opcodes pro Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PLOTS, prefix + 'opcodes_for_label.pdf'))
    plt.clf()

    result_for_labels['opcodes'] = result_for_labels['opcodes'].apply(make_to_general_opcode).apply(
        remove_uninteresting_general_opcodes)
    result_for_general_opcodes = result_for_labels.groupby('opcodes').agg('sum')
    result_for_general_opcodes['opcodes'] = result_for_general_opcodes.index
    result_for_general_plot = result_for_general_opcodes.melt(id_vars=['opcodes'], value_vars=[0, 1, 2]).rename(
        columns={
            'opcodes': 'Opcodes', 'label': 'Cluster', 'value': 'Anzahl',
        })
    ax = sns.barplot(x="Opcodes", y="Anzahl", hue="Cluster", data=result_for_general_plot, palette=[purple_color_3, purple_color_4, purple_color_5])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Anzahl der kategorisierten Opcodes pro Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PLOTS, prefix + 'opcodes_general_for_label.pdf'))
    plt.clf()


def plot_general_opcodes_for_df_with_labels(labeled_general_opcode_df, prefix=''):
    print(labeled_general_opcode_df)
    print_number_of_projects_per_label(labeled_general_opcode_df)
    result_for_labels = labeled_general_opcode_df.drop('project_id', 1).groupby('label').agg(
        'sum').transpose()
    result_for_labels['opcodes'] = result_for_labels.index
    result_for_labels = result_for_labels.melt(id_vars=['opcodes'], value_vars=[0, 1, 2]).rename(columns={
            'opcodes': 'Opcodes', 'label': 'Cluster', 'value': 'Anzahl',
        })
    print(result_for_labels)
    ax = sns.barplot(x="Opcodes", y="Anzahl", hue="Cluster", data=result_for_labels, palette=[purple_color_3, purple_color_4, purple_color_5])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Anzahl der Opcodes pro Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PLOTS, prefix + 'opcodes_general_for_label_general_clustering.pdf'))
    plt.clf()


def print_number_of_projects_per_label(labeled_opcode_df):
    number_of_projects_per_label = labeled_opcode_df.groupby('label')['project_id'].count()
    print(number_of_projects_per_label)
