import argparse
import csv
import json
import logging
import os
import sys
import time

import pandas as pd

from scratch.opcodes import getExistingOpcodesList

from scratch.scratchProject import ScratchProject

parser = argparse.ArgumentParser(description='Read an parse all scratch projects.')
parser.add_argument('projects_folder', type=str, help='Directory containing the scratch projects.')
parser.add_argument('out_folder', type=str, help='Output Directory for the parsed results.')
args = parser.parse_args()

process_all_projects_summary = {
    'opcode_counts': [],
    'number_of_projects': 0
}


def read_all_opcodes(out_folder, projects_folder):
    finished_df = __get_finished_df(out_folder)
    opcodes_file = open(os.path.join(out_folder, 'project_opcodes.csv'), 'w', newline='')
    opcodes_writer = __get_opcodes_writer(opcodes_file)
    start_time = time.time()
    number_of_projects = len(finished_df.index)
    for idx, row in finished_df.iterrows():
        if (idx + 1) % 1000 == 0:
            elapsed_time = time.time() - start_time
            eta = elapsed_time * (number_of_projects / idx)
            logging.info(
                "Number of processed projects: " + str(idx + 1) + "/" + str(number_of_projects) + " Elapsed Time: " +
                str(elapsed_time) + " sec, ETA: " + str(eta) + "sec")
            opcodes_file.flush()
        if row['is_found']:
            process_all_projects_summary['number_of_projects'] += 1
            project = __read_project(os.path.join(projects_folder, str(row['project_id'])+'.json'))
            __write_opcodes(project, opcodes_writer)
    opcodes_file.close()


def __get_finished_df(out_folder):
    return pd.read_csv(os.path.join(out_folder, 'finished.csv'))


def __write_opcodes(project, opcodes_writer):
    counter_opcodes_all, _, counter_opcodes_reduced = project.count_opcodes()

    counter_opcodes_reduced['project_id'] = project.id
    opcodes_writer.writerow(counter_opcodes_reduced)
    number_of_opcodes = sum(counter_opcodes_all.values())
    process_all_projects_summary['opcode_counts'].append(number_of_opcodes)


def __get_opcodes_writer(opcodes_file):
    fieldnames = getExistingOpcodesList()
    fieldnames.insert(0, 'project_id')
    writer = csv.DictWriter(opcodes_file, restval=0, fieldnames=fieldnames)
    writer.writeheader()
    return writer


def __read_project(path_to_project):
    try:
        project = ScratchProject(path_to_project)
    except Exception as error:
        raise Exception("Unsuccessful project: " + path_to_project + ' Reason: Project is not readable ' + str(error))

    if not project.is_valid():
        raise Exception("Unsuccessful project: " + project.id + " Reason: Project has less than 2 targets")

    return project


if __name__ == "__main__":
    logging.basicConfig(filename='./readAllOpcodes.log', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    projects_folder = args.projects_folder
    out_folder = args.out_folder
    logging.info('Current input folder: ' + projects_folder)
    logging.info('Current output folder: ' + out_folder)

    if not os.path.isdir(projects_folder) or not os.path.isdir(out_folder):
        raise Exception('projects_folder or out_folder is not a Directory')
    out_path_projects = os.path.join(out_folder, 'projects')

    read_all_opcodes(out_folder, projects_folder)

    with open(os.path.join(out_folder, 'process_all_projects_summary_opcodes.json'), 'w') as summary_file:
        logging.info(process_all_projects_summary)
        json.dump(process_all_projects_summary, summary_file)
