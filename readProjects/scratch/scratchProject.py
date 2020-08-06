import json
import os

from scratch.monitor import Monitor
from scratch.target import Stage, Sprite
from collections import Counter
from scratch.opcodes import getExistingOpcodesList

existingOpcodes = getExistingOpcodesList()


class ScratchProject:
    def __init__(self, path):
        self.path = path
        self.id = self.__get_id_from_path(path)
        if os.path.isdir(path):
            self.isOnlyJson = False
            project_json, wav_files, svg_files = self.__load_scratch_dir(path)
            self.wavFiles = wav_files
            self.svgFiles = svg_files
        elif os.path.isfile(path) and path.endswith(".json"):
            self.isOnlyJson = True
            self.wavFiles = []
            self.svgFiles = []
            project_json = self.__load_json(path)
        else:
            raise Exception("Scratch Path is not a valid file or dir: " + path)
        self.targets = self.__get_targets(project_json['targets'])
        self.monitors = self.__get_monitors(project_json['monitors'])
        self.extensions = project_json['extensions']
        self.meta = project_json['meta']

    # TODO implement correctly
    def is_valid(self):
        return len(self.targets) > 1

    def get_opcodes(self):
        opcodes = []
        for target in self.targets:
            opcodes.extend(target.get_opcodes())
        return opcodes

    def count_opcodes(self):
        opcodes = self.get_opcodes()
        opcodes_reduced = [opcode for opcode in opcodes if opcode in existingOpcodes]
        opcodes_generalized = []

        for opcode in opcodes:  # mapping all kind of opcode types to one type
            if opcode.startswith('looks'):
                opcodes_generalized.append('looks')
            elif opcode.startswith('control'):
                opcodes_generalized.append('control')
            elif opcode.startswith('data'):
                opcodes_generalized.append('data')
            elif opcode.startswith('motion'):
                opcodes_generalized.append('motion')
            elif opcode.startswith('event'):
                opcodes_generalized.append('event')
            elif opcode.startswith('sound'):
                opcodes_generalized.append('sound')
            else:
                opcodes_generalized.append('other')

        # unordered collection where element is stored as dict key and their count as value
        return Counter(opcodes), Counter(opcodes_generalized), Counter(opcodes_reduced)


    def generate_vector(self):
        opcodes, _ = self.count_opcodes()  # list of opcodes
        vector_opcodes = []
        for opcode in existingOpcodes:
            opcode_count = opcodes.get(opcode)
            if opcode_count is None:
                vector_opcodes.append(0)
            else:
                vector_opcodes.append(opcode_count)
        return vector_opcodes

    def get_comments(self):
        comments = []
        for target in self.targets:
            comments.extend(target.get_comments())
        return comments

    def get_comment_words(self):
        words = []
        for target in self.targets:
            words.extend(target.get_comment_words())
        return words

    def get_comment_word_counts(self):
        word_counts = []
        word_counts_en = []
        word_counts_en_no_stop = []
        for target in self.targets:
            target_word_counts, target_word_counts_en, target_word_counts_en_no_stop = target.get_comment_word_counts()
            word_counts.extend(target_word_counts)
            word_counts_en.extend(target_word_counts_en)
            word_counts_en_no_stop.extend(target_word_counts_en_no_stop)
        return word_counts, word_counts_en, word_counts_en_no_stop

    def __load_scratch_dir(self, path):
        project_json = None
        wav_files = []
        svg_files = []
        files = os.listdir(path)
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(path, file))
            elif file.endswith('.svg'):
                svg_files.append(os.path.join(path, file))
            elif file.endswith('.json'):
                project_json = self.__load_json(os.path.join(path, file))
        if project_json is None:
            raise Exception("Scratch Path does not contain a .json file: " + path)
        return project_json, wav_files, svg_files

    @staticmethod
    def __load_json(json_path):
        with open(json_path, encoding="utf8") as project:
            return json.load(project)

    @staticmethod
    def __get_monitors(monitor_json_list):
        monitor_list = []
        for monitorJson in monitor_json_list:
            monitor_list.append(Monitor(monitorJson))
        return monitor_list

    @staticmethod
    def __get_targets(target_json_list):
        target_list = []
        for target_json in target_json_list:
            if target_json['isStage']:
                target_list.append(Stage(target_json))
            else:
                target_list.append(Sprite(target_json))
        return target_list

    @staticmethod
    def __get_id_from_path(path):
        basename = os.path.basename(path)
        if basename.endswith('.json'):
            basename = basename[:-5]
        return basename
