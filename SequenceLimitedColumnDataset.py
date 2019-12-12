import re
from logging import INFO
from pathlib import Path
from typing import Dict, List, Union

from flair.data import Sentence, Token, Corpus
from flair.datasets import log
from torch.utils.data import Dataset, random_split


class SequenceLimitedColumnCorpus(Corpus):
    def __init__(
            self,
            data_folder: Union[str, Path],
            column_format: Dict[int, str],
            train_file=None,
            test_file=None,
            dev_file=None,
            tag_to_bioes=None,
            in_memory: bool = True,
            max_sequence_length: int = -1,
            stride: int = 4,
            evaluation: bool = False,
            log_level=INFO,
    ):
        """
        Helper function to get a TaggedCorpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.

        :param data_folder: base folder with the task data
        :param column_format: a map specifying the column format
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param tag_to_bioes: whether to convert to BIOES tagging scheme
        :return: a TaggedCorpus with annotated train, dev and test data
        """

        if type(data_folder) == str:
            data_folder: Path = Path(data_folder)

        if train_file is not None:
            train_file = data_folder / train_file
        if test_file is not None:
            test_file = data_folder / test_file
        if dev_file is not None:
            dev_file = data_folder / dev_file

        # automatically identify train / test / dev files
        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if file_name.endswith(".gz"):
                    continue
                if "train" in file_name and not "54019" in file_name:
                    train_file = file
                if "dev" in file_name:
                    dev_file = file
                if "testa" in file_name:
                    dev_file = file
                if "testb" in file_name:
                    test_file = file

            # if no test file is found, take any file with 'test' in name
            if not evaluation and test_file is None:
                for file in data_folder.iterdir():
                    file_name = file.name
                    if file_name.endswith(".gz"):
                        continue
                    if "test" in file_name:
                        test_file = file

        log.log(log_level, "Reading data from {}".format(data_folder))
        log.log(log_level, "Train: {}".format(train_file))
        log.log(log_level, "Dev: {}".format(dev_file))
        log.log(log_level, "Test: {}".format(test_file))

        # get train data
        train = SequenceLimitedColumnDataset(
            train_file, column_format, tag_to_bioes, in_memory=in_memory, max_sequence_length=max_sequence_length,
            stride=stride
        )

        # read in test file if exists, otherwise sample 10% of train data as test dataset
        if test_file is not None:
            test = SequenceLimitedColumnDataset(
                test_file, column_format, tag_to_bioes, in_memory=in_memory, max_sequence_length=max_sequence_length,
                stride=stride
            )
        elif not evaluation:
            train_length = len(train)
            test_size: int = round(train_length / 10)
            splits = random_split(train, [train_length - test_size, test_size])
            train = splits[0]
            test = splits[1]
        else:
            test = Dataset()

        # read in dev file if exists, otherwise sample 10% of train data as dev dataset
        if dev_file is not None:
            dev = SequenceLimitedColumnDataset(
                dev_file, column_format, tag_to_bioes, in_memory=in_memory, max_sequence_length=max_sequence_length,
                stride=stride
            )
        elif not evaluation:
            train_length = len(train)
            dev_size: int = round(train_length / 10)
            splits = random_split(train, [train_length - dev_size, dev_size])
            train = splits[0]
            dev = splits[1]
        else:
            dev = Dataset()

        super(SequenceLimitedColumnCorpus, self).__init__(train, dev, test, name=data_folder.name)


class SequenceLimitedColumnDataset(Dataset):
    def __init__(
            self,
            path_to_column_file: Path,
            column_name_map: Dict[int, str],
            tag_to_bioes: str = None,
            in_memory: bool = True,
            max_sequence_length: int = -1,
            stride: int = 4,
    ):
        """
        :param max_sequence_length: The maximum sequence length. Set to "-1" to disable sequence length check.
        :param stride: The stride on which to split sequences that exceed the maximum length.
        """
        assert path_to_column_file.exists()
        self.path_to_column_file = path_to_column_file
        self.tag_to_bioes = tag_to_bioes
        self.column_name_map = column_name_map

        # store either Sentence objects in memory, or only file offsets
        self.in_memory = in_memory
        if self.in_memory:
            self.sentences: List[Sentence] = []
        else:
            self.indices: List[int] = []

        self.total_sentence_count: int = 0

        # most data sets have the token text in the first column, if not, pass 'text' as column
        self.text_column: int = 0
        self.start_column: int = -1
        self.end_column: int = -1
        for column in self.column_name_map:
            if column_name_map[column] == "text":
                self.text_column = column
            if column_name_map[column] == "begin" or column_name_map[column] == "start":
                self.start_column = column
            if column_name_map[column] == "end":
                self.end_column = column

        # determine encoding of text file
        encoding = "utf-8"
        try:
            lines: List[str] = open(str(path_to_column_file), encoding="utf-8").read(
                10
            ).strip().split("\n")
        except:
            log.info(
                'UTF-8 can\'t read: {} ... using "latin-1" instead.'.format(
                    path_to_column_file
                )
            )
            encoding = "latin1"

        sentence: Sentence = Sentence()
        with open(str(self.path_to_column_file), encoding=encoding) as file:

            split_sentences = 0
            for line in file:
                if line.startswith("#"):
                    continue

                if line.strip().replace("﻿", "") == "":
                    if len(sentence) > 0:
                        sentence.infer_space_after()
                        if self.in_memory:
                            if self.tag_to_bioes is not None:
                                sentence.convert_tag_scheme(
                                    tag_type=self.tag_to_bioes, target_scheme="iobes"
                                )
                            sentence_length = len(" ".join([t.text for t in sentence.tokens]))
                            if 0 < max_sequence_length < sentence_length:
                                split_sentences += 1
                                # # print(f"Found sentence with seq len {sentence_length}", end="")
                                # for i in range(0, int(len(sentence) / stride)):
                                #     split_sentence: Sentence = Sentence()
                                #     offset = i * stride
                                #     curr_len = 0
                                #     for token in sentence.tokens[offset:]:
                                #         if curr_len + (1 + len(token.text)) <= max_sequence_length:
                                #             split_sentence.add_token(token)
                                #             curr_len += len(token.text) + 1
                                #         else:
                                #             break
                                #     self.sentences.append(split_sentence)
                                #     self.total_sentence_count += 1
                            else:
                                self.sentences.append(sentence)
                                self.total_sentence_count += 1
                        else:
                            raise NotImplementedError(
                                "The SequenceLimitedDataset currently only supports in memory operation!")
                    sentence: Sentence = Sentence()

                else:
                    fields: List[str] = re.split("\s+", line.strip())
                    token = self.create_token(fields, column_name_map)
                    sentence.add_token(token)

        if len(sentence.tokens) > 0:
            sentence.infer_space_after()
            if self.in_memory:
                if self.tag_to_bioes is not None:
                    sentence.convert_tag_scheme(
                        tag_type=self.tag_to_bioes, target_scheme="iobes"
                    )
                sentence_length = len(" ".join([t.text for t in sentence.tokens]))
                if 0 < max_sequence_length < sentence_length:
                    split_sentences += 1
                    for i in range(0, int(len(sentence) / stride)):
                        split_sentence: Sentence = Sentence()
                        offset = i * stride
                        curr_len = 0
                        for token in sentence.tokens[offset:]:
                            if curr_len + (1 + len(token.text)) <= max_sequence_length:
                                split_sentence.add_token(token)
                                curr_len += len(token.text) + 1
                            else:
                                break
                        self.sentences.append(split_sentence)
                        self.total_sentence_count += 1
                else:
                    self.sentences.append(sentence)
                    self.total_sentence_count += 1
            else:
                raise NotImplementedError(
                    "The SequenceLimitedDataset currently only supports in memory operation!")

        if split_sentences > 0:
            log.info(f'Split {split_sentences} sentences with a sequence length '
                     f'exceeding {max_sequence_length} characters.')

    def create_token(self, fields: List[Union[int, str]], column_name_map: Dict[int, str]):
        if self.start_column != -1:
            token = Token(fields[self.text_column], start_position=int(fields[self.start_column]))
        else:
            token = Token(fields[self.text_column])
        for column in column_name_map:
            if len(fields) > column:
                if column != self.text_column and column != self.start_column and column != self.end_column:
                    token.add_tag(
                        self.column_name_map[column], fields[column]
                    )
        return token

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:
        if self.in_memory:
            sentence = self.sentences[index]
        else:
            raise NotImplementedError("The SequenceLimitedDataset currently only supports in memory operation!")
        return sentence
