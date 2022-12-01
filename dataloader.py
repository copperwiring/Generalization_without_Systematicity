import torch,os
from torch.utils.data import Dataset


class SCANDatasetLoader(Dataset):
    """Setting up the dataloader"""

    def __read_each_line(filename):
        """Read the file line by line and split into input and output"""
        # Read the file and split into lines
        with open(filename, encoding='utf-8') as file:
            lines = [line.rstrip('\n') for line in file]
            keyword = ' OUT: '
            partition_lines = [line.partition(keyword) for line in
                               lines]  # Split sentences before and after "OUT" keyword
            input_part = [line[0].replace("IN: ", "") for line in partition_lines]
            output_part = [line[2] for line in partition_lines]

        return input_part, output_part

    """the function is initialised here"""

    def __init__(self, root='SCAN', split_type='simple', train=True):
        self.train = train
        self.task = split_type.split("_")[0]
        if self.train:
            self.data_file = os.path.join(root, split_type, f'tasks_train_{self.task}.txt')
        else:
            self.data_file = os.path.join(root, split_type, f'tasks_test_{self.task}.txt')

        print("Loading datafile: ", self.data_file)
        self.input_data, self.output_data = SCANDatasetLoader.__read_each_line(self.data_file)

    def __getitem__(self, index):
        """gives one item at a time"""
        input_data_idx = self.input_data[index]
        output_data_idx = self.output_data[index]

        return input_data_idx, output_data_idx

    def __len__(self):
        """the function returns length of data"""
        return len(self.input_data)


class TokenizerSCAN():
    """
        dataset:SCANDatasetLoader Loaded data with the Torch Dataset
    """

    def __init__(self):
        self.SOS_TOKEN = 0
        self.EOS_TOKEN = 1

        self.word_to_n = {"EOS": self.EOS_TOKEN, "SOS": self.SOS_TOKEN}
        self.n_to_word = dict()
        self.command_to_n = {"EOS": self.EOS_TOKEN, "SOS": self.SOS_TOKEN}
        self.n_to_command = dict()
        self.total_word = 2
        self.total_command = 2

    def get_n_words(self):
        return len(set(self.n_to_word.keys()))

    def get_n_cmds(self):
        return len(set(self.n_to_command.keys()))

    def fit(self, dataset: SCANDatasetLoader):
        for i in range(len(dataset)):
            input_line, output_line = dataset[i]
            for word in input_line.split(" "):
                if word not in self.word_to_n:
                    self.word_to_n[word] = self.total_word
                    self.total_word += 1
            for cmd in output_line.split(" "):
                if cmd not in self.command_to_n:
                    self.command_to_n[cmd] = self.total_command
                    self.total_command += 1
        # Create the Reverse dictionary
        self.n_to_word = {v: k for k, v in self.word_to_n.items()}
        self.n_to_command = {v: k for k, v in self.command_to_n.items()}
        print(f"Total of {self.get_n_words()} words and {self.get_n_cmds()} commands tokenized.")

    def encode_inputs(self, words):
        if type(words) != list:
            if type(words) == str:
                words = words.split(" ")
            else:
                words = list(words)
        return ([self.word_to_n[word] for word in words] + [self.word_to_n["EOS"]] + [self.word_to_n["SOS"]])

    def encode_outputs(self, cmds):
        if type(cmds) != list:
            if type(cmds) == str:
                cmds = cmds.split(" ")
            else:
                cmds = list(cmds)
        return ([self.command_to_n[cmd] for cmd in cmds] + [self.command_to_n["EOS"]])

    def decode_inputs(self, words_n):
        if type(words_n) != list:
            words_n = list(words_n)
        return [self.n_to_word[word_n] for word_n in words_n]

    def decode_outputs(self, cmds_n):
        if type(cmds_n) != list:
            cmds_n = list(cmds_n)
        return [self.n_to_command[cmd_n] for cmd_n in cmds_n]
