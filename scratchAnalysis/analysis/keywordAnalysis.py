import csv
import json
import os

from scratchAnalysis.analysis.textAnalysis import read_all_strings_from_project


def get_projects_with_keywords(keywords, projects_path):
    project_folders = [folder for folder in os.listdir(projects_path) if os.path.isdir(os.path.join(projects_path, folder))]
    projects_with_keywords = {'project_ids': [], 'creation_dates': []}
    for idx, project_folder in enumerate(project_folders):
        project_path = os.path.join(projects_path, project_folder)
        if idx % 10000 == 0:
            print('keywords done: ' + str(idx) + ' of ' + str(len(project_folders)))
        if is_keywords_in_text(keywords, project_path):
            projects_with_keywords['project_ids'].append(project_folder)
            projects_with_keywords['creation_dates'].append(get_creation_date(project_path))
    return projects_with_keywords


def get_all_project_creation_dates(projects_path):
    project_folders = [folder for folder in os.listdir(projects_path) if os.path.isdir(os.path.join(projects_path, folder))]
    creation_dates = {'project_ids': [], 'creation_dates': []}
    for idx, project_folder in enumerate(project_folders):
        project_path = os.path.join(projects_path, project_folder)
        creation_dates['project_ids'].append(project_folder)
        creation_dates['creation_dates'].append(get_creation_date(project_path))
        if idx % 10000 == 0:
            print('keywords done: ' + str(idx) + ' of ' + str(len(project_folders)))
    return creation_dates


def is_keywords_in_text(keywords, project_path):
    texts = read_all_strings_from_project(project_path)
    return any(keyword in str(string).lower() for keyword in keywords for string in texts)


def get_creation_date(project_path):
    project_metainfo = __get_project_metainfo(project_path)
    return project_metainfo['history']['shared']


def __get_project_metainfo(project_path):
    with open(os.path.join(project_path, 'project_metainfo.json')) as metainfo_file:
        return json.load(metainfo_file)