import os
import json
import logging
import csv
import sys
import time

import nltk

from api.scratchApi import get_api_information_for_project
from handleComments.readSentiments import make_sentiment_df_for_comments
from scratch.opcodes import getExistingOpcodesList
from scratch.scratchProject import ScratchProject
import argparse

nltk_download_dir = "/scratch/grassl/python-dependencies/nltk_data"
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')

parser = argparse.ArgumentParser(description='Read an parse all scratch projects.')
parser.add_argument('projects_folder', type=str, help='Directory containing the scratch projects.')
parser.add_argument('out_folder', type=str, help='Output Directory for the parsed results.')
args = parser.parse_args()

process_all_projects_summary = {
    'opcode_counts': [],
    'code_comment_counts': [],
    'code_comment_word_counts': [],
    'project_comment_counts': [],
    'project_comment_word_counts': [],
    'project_creation_dates': [],
    'number_of_projects': 0
}


def process_all_projects(path, out_path_projects, out_path):
    finished_file_path = os.path.join(out_path, 'finished.csv')
    is_existing_finished_file = os.path.isfile(finished_file_path)

    files = __get_files(path, finished_file_path, is_existing_finished_file)
    number_of_projects = len(files)
    logging.info("Number of projects total: " + str(number_of_projects))
    number_of_success = 0
    opcodes_file = open(os.path.join(out_path, 'project_opcodes.csv'), 'w', newline='')
    opcodes_writer = __get_opcodes_writer(opcodes_file)
    finished_file = open(finished_file_path, 'a+', newline='', encoding='utf8')
    finished_writer = __get_finished_writer(finished_file, is_existing_finished_file)

    start_time = time.time()
    for idx, file in enumerate(files):
        if (idx + 1) % 1000 == 0:
            elapsed_time = time.time() - start_time
            eta = elapsed_time * (number_of_projects / idx)
            logging.info(
                "Number of processed projects: " + str(idx + 1) + "/" + str(number_of_projects) + " Elapsed Time: " +
                str(elapsed_time) + " sec, ETA: " + str(eta) + "sec")
            opcodes_file.flush()
            finished_file.flush()
        try:
            project_id = file[:-5]
            process_single_project(project_id, os.path.join(path, file), out_path_projects, opcodes_writer,
                                   finished_writer)
            number_of_success += 1
            logging.debug("Successful project: " + project_id)
        except Exception as error:
            logging.info(error)
    logging.info("Total Number of successful projects " + str(number_of_success))
    opcodes_file.close()
    finished_file.close()


def process_single_project(project_id, path_to_project, out_path, opcodes_writer, finished_writer):
    is_found = False
    project_metainfo = dict()
    try:
        project = __read_project(path_to_project)

        code_comments = project.get_comments()
        summary_code_comments, df_code_comments = make_sentiment_df_for_comments(code_comments)
        project_metainfo, project_comments = get_api_information_for_project(project.id)
        # Write Opcodes after the project is fetched via the api, as it still exists.
        is_found = True
        __write_opcodes(project, opcodes_writer)
        process_all_projects_summary['number_of_projects'] += 1
        process_all_projects_summary['project_creation_dates'].append(project_metainfo['history']['created'])

        summary_project_comments, df_project_comments = make_sentiment_df_for_comments(project_comments)
        if summary_code_comments['number_of_comments'] == 0 and summary_project_comments['number_of_comments'] == 0:
            project_out_folder = os.path.join(out_path, project.id)
            __handle_comments(project_out_folder, project_metainfo, summary_code_comments, df_code_comments,
                              summary_project_comments, df_project_comments)

    except Exception as error:
        logging.debug(error)

    __write_finished(project_id, is_found, project_metainfo, finished_writer)


def __read_project(path_to_project):
    try:
        project = ScratchProject(path_to_project)
    except Exception as error:
        raise Exception("Unsuccessful project: " + path_to_project + ' Reason: Project is not readable ' + str(error))

    if not project.is_valid():
        raise Exception("Unsuccessful project: " + project.id + " Reason: Project has less than 2 targets")

    return project


def __write_comment_out_files(out_path, summary, df_comments, comment_type):
    with open(os.path.join(out_path, comment_type + '_summary.json'), 'w') as summary_out:
        json.dump(summary, summary_out)
    if summary['number_of_comments'] > 0:
        df_comments.to_csv(os.path.join(out_path, 'all_' + comment_type + '.csv'))


def __write_opcodes(project, opcodes_writer):
    counter_opcodes_all, _, counter_opcodes_reduced = project.count_opcodes()

    counter_opcodes_reduced['project_id'] = project.id
    opcodes_writer.writerow(counter_opcodes_reduced)

    number_of_opcodes = sum(counter_opcodes_all.values())
    process_all_projects_summary['opcode_counts'].append(number_of_opcodes)


def __handle_comments(project_out_folder, project_metainfo, summary_code_comments, df_code_comments,
                      summary_project_comments, df_project_comments):
    if not os.path.exists(project_out_folder):
        os.makedirs(project_out_folder)

    with open(os.path.join(project_out_folder, 'project_metainfo.json'), 'w') as project_metainfo_out:
        json.dump(project_metainfo, project_metainfo_out)

    __write_comment_out_files(project_out_folder, summary_code_comments, df_code_comments, 'code_comments')
    __write_comment_out_files(project_out_folder, summary_project_comments, df_project_comments, 'project_comments')

    process_all_projects_summary['code_comment_counts'].append(summary_code_comments['number_of_comments'])
    process_all_projects_summary['code_comment_word_counts'].append(summary_code_comments['number_of_words'])
    process_all_projects_summary['project_comment_counts'].append(summary_project_comments['number_of_comments'])
    process_all_projects_summary['project_comment_word_counts'].append(summary_project_comments['number_of_words'])


def __write_finished(project_id, is_found, project_metainfo, finished_writer):
    metainfo = json.dumps({'title': project_metainfo.get('title'),
                           'description': project_metainfo.get('description'),
                           'instructions': project_metainfo.get('instructions')})
    row = {'project_id': project_id,
           'is_found': is_found,
           'metainfo': metainfo}
    finished_writer.writerow(row)


def __get_opcodes_writer(opcodes_file):
    fieldnames = getExistingOpcodesList()
    fieldnames.insert(0, 'project_id')
    writer = csv.DictWriter(opcodes_file, restval=0, fieldnames=fieldnames)
    writer.writeheader()
    return writer


def __get_finished_writer(finished_file, is_existing_file):
    fieldnames = ['project_id', 'is_found', 'metainfo']
    writer = csv.DictWriter(finished_file, fieldnames=fieldnames)
    if not is_existing_file:
        writer.writeheader()
    return writer


def __get_files(path, finished_file_path, is_existing_finished_file):
    finished_files = []
    if is_existing_finished_file:
        with open(finished_file_path, newline='', encoding='utf8') as finished_file:
            reader = csv.DictReader(finished_file)
            for row in reader:
                finished_files.append(row['project_id'] + '.json')
    logging.info("Number of finished: " + str(len(finished_files)))
    return [file for file in os.listdir(path) if file not in finished_files]


if __name__ == "__main__":
    logging.basicConfig(filename='./readAllProjects.log', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    projects_folder = args.projects_folder
    out_folder = args.out_folder
    logging.info('Current input folder: ' + projects_folder)
    logging.info('Current output folder: ' + out_folder)

    if not os.path.isdir(projects_folder) or not os.path.isdir(out_folder):
        raise Exception('projects_folder or out_folder is not a Directory')
    out_path_projects = os.path.join(out_folder, 'projects')

    if not os.path.isdir(out_path_projects):
        os.mkdir(out_path_projects)

    logging.info('Start processing all projects:')
    process_all_projects(projects_folder, out_path_projects, out_folder)

    with open(os.path.join(out_folder, 'process_all_projects_summary.json'), 'w') as summary_file:
        json.dump(process_all_projects_summary, summary_file)
