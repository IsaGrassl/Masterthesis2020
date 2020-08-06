import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
import pandas as pd

sns.set(style='whitegrid', palette=sns.cubehelix_palette(8, reverse=True))

OUT_PLOTS = "C:/Users/Isabella/Documents/Masterthesis/Code/out/plots/kmeans"


def get_lables_from_kmeans(opcodes_matrix):
    # creating classifier and training
    k_means = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(opcodes_matrix)
    k_means.predict(opcodes_matrix)
    labels = k_means.labels_
    return labels


def clustering_opcodes(opcodes_df, out_prefix, norm, labels=None):
    current_out_path = os.path.join(OUT_PLOTS, out_prefix)
    if not os.path.isdir(current_out_path):
        os.mkdir(current_out_path)

    # Normalize Opcodes
    opcodes_df_without_id = opcodes_df.drop('project_id', 1)
    print(opcodes_df_without_id)
    transformer = Normalizer(norm=norm).fit(opcodes_df_without_id)
    opcodes_matrix = transformer.transform(opcodes_df_without_id)
    print(opcodes_matrix)

    # creating classifier and training
    if labels is None:
        labels = get_lables_from_kmeans(opcodes_matrix)
    opcodes_df['label'] = labels

    # normalized_df = pd.DataFrame(data=opcodes_matrix, index=opcodes_df_without_id.index, columns=opcodes_df_without_id.columns )
    # normalized_df['label'] = labels
    #centers = k_means.cluster_centers_

    silhouette = silhouette_score(opcodes_matrix, labels)
    calinski = calinski_harabasz_score(opcodes_matrix, labels)
    print("The average silhouette_score is :", silhouette)
    print("The average calinski_harabasz_score is :", calinski)

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(opcodes_matrix)
    results = pd.DataFrame(reduced_data, columns=['Komponente 1', 'Komponente 2'])
    sns.pairplot(results, diag_kind='kde')
    plt.tight_layout()
    plt.savefig(os.path.join(current_out_path, 'pairplot_pca.pdf'))
    plt.clf()

    df = pd.DataFrame(pca.components_, columns=opcodes_df_without_id.columns,
                      index=['Komponente 1', 'Komponente 2'])
    print(df)
    df.to_csv(os.path.join(current_out_path, 'pca_komponents_df.csv'))

    explained_variance = np.var(reduced_data, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    cumulative_proportion_variance = np.cumsum(
        explained_variance_ratio)  # ratio between variance of PC and total variance
    # we see how much variance each variable explains
    # print(cumulative_proportion_variance)

    ax = sns.scatterplot(x="Komponente 1", y="Komponente 2", hue=labels, data=results)
    # ax.scatter(centers[:,0], centers[:,1], marker="x", s=150.0, color="purple")
    ax.set_xlabel('Komponente 1 (%.2f%%)' % (explained_variance_ratio[0] * 100))
    ax.set_ylabel('Komponente 2 (%.2f%%)' % (explained_variance_ratio[1] * 100))
    ax.set_title('Clustering mit zwei Dimensionen')
    plt.tight_layout()
    plt.savefig(os.path.join(current_out_path, 'clustering_pca_with_two_dimensions.png'))
    plt.savefig(os.path.join(current_out_path, 'clustering_pca_with_two_dimensions.pdf'))
    plt.clf()

    pca = PCA().fit(opcodes_matrix)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    print(pca.components_)
    plt.xlabel('Anzahl an Komponenten')
    plt.ylabel('Cumulative explained variance')
    plt.title('Anzahl an Komponenten, die nötig sind für die Beschreibung der Daten')
    plt.tight_layout()
    plt.savefig(os.path.join(current_out_path, 'necessary_components.pdf'))
    plt.clf()
    return opcodes_df


# estimate optimal number of clusters
def elbow_opcodes(opcodes_df):
    opcodes_df_without_id = opcodes_df.drop('project_id', 1)
    transformer = Normalizer().fit(opcodes_df_without_id)
    opcodes_matrix = transformer.transform(opcodes_df_without_id)

    distortion = []
    for i in range(1, 8):
        k_means = KMeans(n_clusters=i, init='random', n_init=10, max_iter=300, random_state=0).fit(
            opcodes_matrix)
        distortion.append(k_means.inertia_)
    # plt.plot(distortion, marker='o')

    data = {"Distortion": distortion, "Anzahl der Cluster": range(1, 8)}
    print(distortion)
    ax = sns.lineplot(x="Anzahl der Cluster", y="Distortion", ci=None, data=data, markers=True)
    ax.set_title('Elbow-Kriterium')
    ax.set_xlabel('Anzahl der Cluster')
    ax.set_ylabel('Distortion')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PLOTS, 'elbow_plot.pdf'))
    plt.clf()


# TODO not working atm
def dbscan_opcodes(opcodes_matrix, all_projects):
    dbscan = DBSCAN(eps=0.2, min_samples=2).fit(opcodes_matrix)
    labels = dbscan.labels_
    print(labels)
    reduced_data = PCA(n_components=2).fit_transform(opcodes_matrix)
    results = pd.DataFrame(reduced_data, columns=['pca1', 'pca2'])
    sns.scatterplot(x="pca1", y="pca2", hue=dbscan, data=results)
    plt.title('DBSCAN Clustering mit zwei Dimensionen')
    plt.show()
    label_color = {label: idx for idx, label in enumerate(np.unique(labels))}
    colorvec = [label_color[label] for label in labels]
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colorvec, edgecolor='', alpha=0.5)
    plt.title('DBSCAN Clustering mit fünf Clustern in zwei Dimension')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def correlationmatrix(opcodes_df):
    corrmatrix = opcodes_df.drop('project_id', 1).corr()
    print(corrmatrix)
    plt.matshow(corrmatrix)
    plt.xticks(range(corrmatrix.shape[1]), corrmatrix.columns, fontsize=14, rotation=45)
    plt.yticks(range(corrmatrix.shape[1]), corrmatrix.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Korrelationsmatrix')
    plt.savefig(os.path.join(OUT_PLOTS, 'correlation_matrix_opcodes.pdf'))
    plt.clf()
    sns.heatmap(corrmatrix, annot=True)
    plt.title('HeatMap')
    plt.savefig(os.path.join(OUT_PLOTS, 'heat_map_opcodes.pdf'))
    plt.clf()


def load_opcodes_df(projects_out):
    return pd.read_csv(os.path.join(projects_out, 'project_opcodesV2.csv'))


def make_opcodes_general_df(opcodes_df):
    project_ids = opcodes_df['project_id']
    transposed = opcodes_df.drop('project_id', 1).transpose()
    transposed['opcodes'] = transposed.index
    transposed['opcodes'] = transposed['opcodes'].apply(make_to_general_opcode).apply(
        remove_uninteresting_general_opcodes)
    result = transposed.groupby('opcodes').agg('sum').transpose()
    result['project_id'] = project_ids
    return result


def make_to_general_opcode(opcode):
    return opcode.split('_')[0]


def remove_uninteresting_general_opcodes(opcode):
    uninteresting_general_opcodes = ['argument', 'operator', 'sensing']
    if opcode in uninteresting_general_opcodes:
        return None
    else:
        return opcode
