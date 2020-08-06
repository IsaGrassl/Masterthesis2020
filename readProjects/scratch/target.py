import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from langdetect import detect

stops = stopwords.words('english')


class Target:
    def __init__(self, json):
        # True if this is the stage and false otherwise.
        self.isStage = json.get('isStage')
        # The name.
        self.name = json.get('name')
        # An object associating IDs with arrays representing variables
        # whose first element is the variable's name followed by it's value.
        self.variables = json.get('variables')
        # An object associating IDs with arrays representing lists
        # whose first element is the list's name followed by the list as an array.
        self.lists = json.get('lists')
        # An object associating IDs with broadcast names.
        self.broadcasts = json.get('broadcasts')
        # An object associating IDs with blocks.
        self.blocks = self.__create_blocks((json.get('blocks')))
        # An object associating IDs with comments.
        self.comments = self.__create_comments(json.get('comments'))
        # The costume number.
        self.currentCostume = json.get('currentCostume')
        # An array of costumes.
        self.costumes = json.get('costumes')
        # An array of sounds.
        self.sounds = json.get('sounds')
        # The volume.
        self.volume = json.get('volume')
        # The layer number.
        self.layerOrder = json.get('layerOrder')

    @staticmethod
    def __create_blocks(block_json_dict):
        block_list = []
        for block_id, block_json in block_json_dict.items():
            if block_json.get('topLevel'):
                block_list.append(Block(block_id, block_json_dict, None))
        return block_list

    def __create_comments(self, comments_json_dict):
        comment_list = []
        for comment_id, comment_json in comments_json_dict.items():
            comment_list.append(Comment(comment_id, comment_json, self))
        return comment_list

    def get_opcodes(self):
        opcodes = []
        for block in self.blocks:
            opcodes.extend(block.get_opcodes())
        return opcodes

    def get_block_by_id(self, block_id):
        for block in self.blocks:
            tmp_block = block.get_block_by_id(block_id)
            if tmp_block is not None:
                return tmp_block
        return None

    def get_comments(self):
        return [x.get_comment_text() for x in self.comments]

    def get_comment_words(self):
        words = []
        for comment in self.comments:
            words.extend(comment.get_filtered_text())
        return words

    def get_comment_word_counts(self):
        word_counts = []
        word_counts_en = []
        word_counts_en_no_stop = []
        for comment in self.comments:
            word_count, word_count_en_no_stop = comment.get_word_count()
            word_counts.append(word_count)
            if word_count_en_no_stop != 0:
                word_counts_en.append(word_count)
                word_counts_en_no_stop.append(word_count_en_no_stop)
        return word_counts, word_counts_en, word_counts_en_no_stop


class Stage(Target):
    def __init__(self, json):
        Target.__init__(self, json)
        # The tempo in BPM.
        self.tempo = json.get('tempo')
        # The video transparency.
        self.videoTransparency = json.get('videoTransparency')
        # If video sensing has been turned off and is off, this is "off";
        # if the video is flipped, it is "on-flipped"; otherwise, it is "on".
        self.videoState = json.get('videoState')
        # The language of the Text to Speech extension.
        self.textToSpeechLanguage = json.get('textToSpeechLanguage')


class Sprite(Target):
    def __init__(self, json):
        Target.__init__(self, json)
        # True if the monitor is visible and false otherwise.
        self.visible = json.get('visible')
        # The x - coordinate.
        self.x = json.get('x')
        # The y - coordinate.
        self.y = json.get('y')
        # The sprite's scaling factor as a percentage.
        self.size = json.get('size')
        # The sprite's direction in degrees clockwise from North.
        self.direction = json.get('direction')
        # True if the sprite is draggable and false otherwise.
        self.draggable = json.get('draggable')
        # The name of the rotation style: "all around", "left-right", or "don't rotate".
        self.rotationStyle = json.get('rotationStyle')


class Block:
    def __init__(self, block_id, all_target_blocks, parent):
        json = all_target_blocks[block_id]
        # Id given via the blocks json object in the target
        self.id = block_id
        # A string naming the block. The opcode of a "core" block may be found in the Scratch source code.
        self.opcode = json.get('opcode')
        # The ID of the following block or null.
        self.next = self.__create_block_for_id(json.get('next'), all_target_blocks)
        # If the block is a stack block and is preceded, this is the ID of the preceding block.
        # If the block is the first stack block in a C mouth, this is the ID of the C block.
        # If the block is an input to another block, this is the ID of that other block. Otherwise it is null.
        self.parent = parent
        # An object associating names with arrays representing inputs into which reporters may be dropped and C mouths.
        # The first element of each array is 1 if the input is a shadow, 2 if there is no shadow,
        # and 3 if there is a shadow but it is obscured by the input.
        # The second is either the ID of the input or an array representing it as described below.
        # If there is an obscured shadow, the third element is its ID or an array representing it.
        self.inputs = json.get('inputs')
        # An object associating names with arrays representing fields.
        # The first element of each array is the field's value which may be followed by an ID.
        self.fields = json.get('fields')
        #  True if this is a shadow and false otherwise.
        self.shadow = json.get('shadow')
        # False if the block has a parent and true otherwise.
        self.topLevel = json.get('topLevel')

    def __create_block_for_id(self, block_id, all_target_blocks):
        if block_id:
            return Block(block_id, all_target_blocks, self)
        else:
            return None

    def get_block_by_id(self, block_id):
        if self.id is block_id:
            return self
        elif self.next:
            return self.next.get_block_by_id(block_id)
        else:
            return None

    def get_opcodes(self):
        if self.next is None:
            return [self.opcode]  #
        else:
            opcodes = self.next.get_opcodes()
            opcodes.append(self.opcode)
            return opcodes


class Comment:
    def __init__(self, comment_id, json, target):
        self.id = comment_id
        # The ID of the block the comment is attached to.
        self.blockId = json.get('blockId')
        self.block = target.get_block_by_id(self.blockId)
        # The x-coordinate of the comment in the code area.
        self.x = json.get('x')
        # The y-coordinate.
        self.y = json.get('y')
        # The width.
        self.width = json.get('width')
        # The height.
        self.height = json.get('height')
        # True if the comment is collapsed and false otherwise.
        self.minimized = json.get('minimized')
        # The text.
        self.text = json.get('text')

    def get_comment_text(self):
        return self.text

    def get_filtered_text(self):
        tokenized = word_tokenize(self.text)
        lang = ''
        try:
            lang = detect(self.text)
        except:
            pass
        if lang == 'en':
            filtered_words = [word.lower() for word in tokenized if word.lower() not in stops and word.isalpha()]
            return filtered_words
        return []

    def get_word_count(self):
        word_count = len(word_tokenize(self.text))
        word_count_en_no_stop = len(self.get_filtered_text())
        return word_count, word_count_en_no_stop
