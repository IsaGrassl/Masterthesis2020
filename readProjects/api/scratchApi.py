import time

import requests
import os
import json
import logging

STORED_REQUESTS_DIR = "C:/Users/Isabella/Documents/Masterthesis/Code/out/storedRequests"
PROJECT_URL = "https://api.scratch.mit.edu/projects/{0}"
USER_URL = "https://api.scratch.mit.edu/users/{0}"
ALL_PROJECTS_FOR_USER_URL = "https://api.scratch.mit.edu/users/{0}/projects"
COMMENTS_FOR_PROJECT_URL = "https://api.scratch.mit.edu/users/{0}/projects/{1}/comments"
COMMENT_REPLY_FOR_PROJECT_URL = "https://api.scratch.mit.edu/users/{0}/projects/{1}/comments/{2}/replies"
SHOULD_STORE_REQUESTS = False


def get_project(project_id):
    url = PROJECT_URL.format(project_id)
    request_dir = os.path.join(STORED_REQUESTS_DIR, project_id)
    return __get_stored_or_get_request(url, request_dir, 'project_request.json')


def get_user(username, project_id):
    url = USER_URL.format(username)
    request_dir = os.path.join(STORED_REQUESTS_DIR, project_id)
    return __get_stored_or_get_request(url, request_dir, 'user_request.json')


def get_comments_for_project(username, project_id):
    url = COMMENTS_FOR_PROJECT_URL.format(username, project_id)
    request_dir = os.path.join(STORED_REQUESTS_DIR, project_id)
    return __get_stored_or_get_request(url, request_dir, 'project_comments_request.json')


def get_replies_for_comment(username, project_id, comment_id):
    url = COMMENT_REPLY_FOR_PROJECT_URL.format(username, project_id, comment_id)
    request_dir = os.path.join(STORED_REQUESTS_DIR, project_id)
    return __get_stored_or_get_request(url, request_dir, 'comment_{0}_replies_request.json'.format(comment_id))


def get_all_comments_for_project(project_id):
    return []


def get_api_information_for_project(project_id):
    if SHOULD_STORE_REQUESTS and not os.path.isdir(STORED_REQUESTS_DIR):
        os.mkdir(STORED_REQUESTS_DIR)
    project = __get_project_with_multiple_tries(project_id)
    if project.get('code') == 'NotFound':
        raise Exception('Project not found')
    comments = get_comments_for_project(project['author']['username'], project_id)
    comment_contents = []
    for comment in comments:
        comment_contents.append(comment['content'])
        if comment['reply_count'] > 0:
            replies = get_replies_for_comment(project['author']['username'], project_id, comment['id'])
            for reply in replies:
                comment_contents.append(reply['content'])
    return project, comment_contents


def __get_project_with_multiple_tries(project_id):
    number_of_tries = 0
    error = Exception("Number of tries exceeded")
    while number_of_tries < 3:
        try:
            number_of_tries += 1
            return get_project(project_id)
        except Exception as e:
            error = e
            logging.warning('Fetch from Api did not work for Project: ' + project_id + ' Reason: ' + str(e))
            time.sleep(15)
    raise error


def __get_stored_or_get_request(url, request_dir, request_file_name):
    if SHOULD_STORE_REQUESTS:
        request_file_path = os.path.join(request_dir, request_file_name)
        if os.path.isdir(request_dir):
            if os.path.isfile(request_file_path):
                with open(request_file_path) as request_file:
                    return json.load(request_file)
        else:
            os.mkdir(request_dir)
        response_json = requests.get(url).json()
        with open(request_file_path, 'w') as outfile:
            json.dump(response_json, outfile)
    else:
        response_json = requests.get(url).json()
    return response_json
