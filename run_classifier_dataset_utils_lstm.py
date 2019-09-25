# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
import pickle

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score


logger = logging.getLogger(__name__)



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        #import pdb; pdb.set_trace()
        csv.field_size_limit(sys.maxsize)

        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class PatentTS36Processor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            if len(line) != 3:
                import pdb; pdb.set_trace()

            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

class USPTOWirelessProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self):
        """See base class."""

        return ['H04R', 'H04N', 'H04B', 'H04L', 'H04Q', 'H04M', 'H04J', 'H04K', 'H04H', 'H04S', 'H04W']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1] 
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

class USPTOAgProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self):
        """See base class."""

        return ['A01N', 'A01K', 'A01H', 'A01D', 'A01B', 'A01G', 'A01M', 'A01F', 'A01C', 'A01J', 'A01L']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1] 
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

class USPTOChemProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self):
        """See base class."""

        return ['B01L', 'B01J', 'B01B', 'B01D', 'B01F']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1] 
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples
        
class USPTOComputingProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self):
        """See base class."""

        return ['G06G', 'G06K', 'G06D', 'G06F', 'G06J', 'G06N', 'G06M', 'G06C', 'G06E', 'G06T', 'G06Q']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1] 
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

class USPTOAllSubClassProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self):
        """See base class."""

        return ['A01N', 'B31B', 'F02K', 'H03L', 'F01N', 'A23J', 'A44B', 'F16N', 'G04F', 'D06G', 'E04H', 'F02G', 'G01T', 'H02S', 'C07J', 'B43K', 'B03D', 'G06E', 'B26F', 'H03K', 'H03J', 'C11B', 'H01G', 
        'B66B', 'B29K', 'F23J', 'A47F', 'G01K', 'F04D', 'B23C', 'A63B', 'B63B', 'C13K', 'C07K', 'D21G', 'E21B', 'F04F', 'C01F', 'B28C', 'C05F', 'C09D', 'A41G', 'B82B', 'A61G', 'C21B', 'H04S', 'A23V', 
        'H05K', 'F05D', 'A23P', 'G06G', 'F23G', 'A61N', 'C25B', 'H05F', 'F02P', 'C22F', 'E01D', 'G09D', 'B61C', 'A21C', 'H03C', 'H01K', 'G08C', 'B63C', 'C05D', 'B61J', 'B60S', 'A01H', 'G21C', 'F03G', 
        'D01B', 'E02D', 'E01F', 'C21C', 'A62C', 'F22D', 'C10C', 'F24C', 'B42C', 'A23B', 'B25B', 'G10H', 'C08G', 'H01M', 'A63G', 'B42B', 'C12G', 'G01R', 'B25D', 'F03B', 'A41H', 'C10N', 'C23D', 'B29C', 
        'D03J', 'A62B', 'B60T', 'B41C', 'B21B', 'H03G', 'B61G', 'D21C', 'A01K', 'A42B', 'B26D', 'B25G', 'E03F', 'B65G', 'C07H', 'F02D', 'F23C', 'A41B', 'B60P', 'D05C', 'C12Q', 'G03D', 'G10G', 'B68G', 
        'H04L', 'E02B', 'G04R', 'D04C', 'C12M', 'D06P', 'B27F', 'C08B', 'B64B', 'F22B', 'G06Q', 'G05G', 'A47C', 'D06B', 'F02M', 'F15D', 'A63J', 'H01R', 'A61K', 'A61H', 'G04B', 'G01F', 'B05C', 'B81B', 
        'B42P', 'B27N', 'F17B', 'B66D', 'E04C', 'G03B', 'H01J', 'C10L', 'H01F', 'A47L', 'B64D', 'H01T', 'B27L', 'A23F', 'A63H', 'F27D', 'C11C', 'B41G', 'A61D', 'C10J', 'F16K', 'F16F', 'C07C', 'B61L', 
        'B21C', 'B65C', 'F23K', 'C01B', 'H02B', 'A43B', 'G21D', 'D04G', 'C06B', 'G09G', 'G03F', 'G21F', 'B21F', 'G01V', 'A44C', 'F41F', 'G01G', 'D21H', 'D06Q', 'B65D', 'A43C', 'G04D', 'G07G', 'C03C', 
        'D04H', 'H04H', 'B08B', 'A01D', 'F28G', 'C03B', 'A62D', 'C09J', 'B67C', 'E04D', 'C10F', 'A61F', 'E01C', 'G06T', 'G10C', 'F17D', 'B62H', 'B23F', 'B62C', 'F02N', 'B60L', 'B05B', 'H03B', 'D10B', 
        'A63F', 'F23D', 'B65F', 'C05C', 'H03F', 'G10K', 'C12C', 'F41H', 'G21H', 'B04C', 'H05C', 'G04C', 'B27M', 'G21B', 'A01J', 'A45D', 'F02B', 'A21B', 'F03H', 'B23H', 'B23Q', 'B82Y', 'C04B', 'D05D', 
        'B21D', 'A01B', 'D04B', 'C40B', 'F01P', 'E21F', 'A47J', 'B62D', 'C11D', 'B27H', 'B60V', 'G02F', 'B27K', 'H03H', 'F28C', 'D21F', 'G01D', 'B60G', 'F42D', 'E04B', 'F23N', 'B41M', 'A23N', 'E05B', 
        'F02F', 'G01J', 'H01P', 'E05G', 'G11C', 'B03C', 'E05C', 'A01G', 'B44C', 'B62K', 'F24H', 'A24C', 'E06B', 'C09K', 'E03C', 'D06H', 'G05B', 'A41C', 'F21Y', 'F21L', 'F23R', 'A45C', 'B22C', 'A45B', 
        'B07C', 'B60D', 'B09B', 'C25D', 'F16B', 'F24F', 'C23F', 'B60B', 'B31D', 'B01B', 'B61F', 'B44D', 'A23K', 'B65B', 'F23L', 'F28B', 'G10B', 'D01D', 'B21L', 'F15C', 'B33Y', 'B28B', 'F25C', 'A41F', 
        'A42C', 'C13B', 'C01C', 'F16T', 'B01D', 'E05D', 'F21W', 'B43L', 'C01G', 'B29L', 'B41J', 'A63D', 'H04R', 'B23G', 'B61B', 'C09C', 'F21S', 'F24D', 'B60Y', 'E06C', 'G05F', 'G21K', 'C05B', 'F01K', 
        'C08K', 'F16C', 'B03B', 'F01D', 'B01L', 'C07F', 'B44F', 'C09G', 'B42F', 'F23Q', 'B29D', 'C23C', 'B60N', 'B41D', 'E04F', 'B23K', 'H02K', 'C09H', 'B31F', 'F16H', 'E01H', 'B07B', 'H01B', 'C14C', 
        'H01S', 'D21J', 'B66F', 'B67D', 'H01C', 'C08L', 'B27D', 'D03C', 'F16G', 'B68B', 'A45F', 'F01M', 'C01D', 'C08H', 'C12F', 'D06L', 'B32B', 'D02J', 'C07B', 'A41D', 'H05B', 'D21D', 'B64C', 'B43M', 
        'G07F', 'H05H', 'B61H', 'C06D', 'B41B', 'B30B', 'B25C', 'D06J', 'A22B', 'C08J', 'B60F', 'H04B', 'B60W', 'A63K', 'G08G', 'H04K', 'H04N', 'F28F', 'G06J', 'C09B', 'G03H', 'B27C', 'A61C', 'G07D', 
        'C25F', 'B27G', 'F23B', 'C06C', 'G10F', 'B24B', 'B05D', 'B62L', 'C05G', 'H02N', 'C23G', 'C12J', 'H02H', 'F41J', 'A47B', 'G08B', 'G01P', 'F16M', 'D06F', 'D06M', 'B22F', 'B09C', 'G01C', 'C12P', 
        'D01C', 'E21D', 'B41F', 'B23P', 'A61B', 'F01B', 'G10L', 'B62M', 'C10G', 'G03G', 'B67B', 'B01J', 'G01H', 'B44B', 'G01N', 'F26B', 'A47H', 'B21G', 'B41K', 'B25H', 'G09B', 'D07B', 'F05C', 'H01H', 
        'A47K', 'B42D', 'G12B', 'B25J', 'A23Y', 'B62B', 'B66C', 'F41G', 'B04B', 'B23B', 'C01P', 'F16P', 'H02M', 'F22G', 'E01B', 'A44D', 'E05Y', 'C22B', 'F05B', 'C25C', 'B41L', 'B65H', 'F25J', 'G01W', 
        'C12L', 'H03M', 'G10D', 'G06F', 'C14B', 'A21D', 'E21C', 'B22D', 'C10M', 'B61D', 'D04D', 'B24D', 'A46D', 'H02G', 'D03D', 'G01Q', 'G03C', 'B63J', 'A61J', 'H01L', 'F24J', 'A63C', 'B27B', 'F41B', 
        'D21B', 'B29B', 'B60C', 'G01B', 'F41A', 'B41P', 'B41N', 'A01L', 'C12H', 'D01H', 'B24C', 'B23D', 'B31C', 'F27B', 'B60K', 'G01M', 'G06N', 'G09F', 'F16J', 'B27J', 'E02F', 'F23M', 'C12N', 'B81C', 
        'H03D', 'A24F', 'A61L', 'A23G', 'D01F', 'D02H', 'C10H', 'D02G', 'B02C', 'E03D', 'G01S', 'C21D', 'G02B', 'B60R', 'A47D', 'C08C', 'E02C', 'B62J', 'C07D', 'B63H', 'A23C', 'F23H', 'C08F', 'H01Q', 
        'F15B', 'C30B', 'F03C', 'E04G', 'A24B', 'H04J', 'B64F', 'C10K', 'A61M', 'D06N', 'E03B', 'G21G', 'A22C', 'G06K', 'F17C', 'B61K', 'C02F', 'H04Q', 'A23L', 'G07C', 'F01C', 'B26B', 'C12R', 'A24D', 
        'G05D', 'F21V', 'A01F', 'F16D', 'F16L', 'A61Q', 'F42C', 'D06C', 'B06B', 'B63G', 'F24B', 'B02B', 'B21H', 'H02P', 'G04G', 'F01L', 'A43D', 'B01F', 'G02C', 'G06C', 'A46B', 'C22C', 'B25F', 'C10B', 
        'F41C', 'A47G', 'B68C', 'F25D', 'A01M', 'F21H', 'F42B', 'G01L', 'B60Q', 'G06M', 'G07B', 'F28D', 'F04C', 'B60J', 'E05F', 'B64G', 'F03D', 'B21J', 'G09C', 'F25B', 'F04B', 'F21K', 'G11B', 'D05B', 
        'C07G', 'H05G', 'H02J', 'A23D', 'H04W', 'A01C', 'B28D', 'B21K', 'B60M', 'B60H', 'C12Y', 'D01G', 'F02C', 'C09F', 'H04M']


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1] 
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

class USPTOAllSectionProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ['A','B','C','D','E','F','G','H']


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1] 
            label = line[2][0] #PARSE THE ALLSUBCLASS DATASET, BUT ONLY GRAB THE SECTIONS
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class ArxivProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ['cs.CE',  'cs.DS', 'cs.NE',  'cs.SY',  'math.AC',  'math.ST', 'cs.AI', 'cs.cv', 'cs.IT', 'cs.PL',  'math.GR']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1] 
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, seq_segments,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label : i for i, label in enumerate(label_list)}
    
    tokenizer.max_len = total_input_length = (max_seq_length * seq_segments)

    extra_tokens = 2* (total_input_length // max_seq_length)
    tokenless_total_len = total_input_length - extra_tokens


    features = []
    for (ex_index, example) in enumerate(examples):
        
        if ex_index % 500 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            exit("cant handle b tokens right now")
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, total_input_length - 3)
        else:
            if len(tokens_a) > tokenless_total_len :
                tokens_a = tokens_a[:tokenless_total_len ]


        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        i = 254
        while i < len(tokens_a):
            tokens_a.insert(i,"[CLS]")
            tokens_a.insert(i,"[SEP]")
            i+=256

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

       
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (total_input_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == total_input_length
        assert len(input_mask) == total_input_length
        assert len(segment_ids) == total_input_length

        if output_mode == "classification":
            if example.label not in label_list:
                print("bad label on example {}, {}".format(ex_index,example.guid))
                continue
            label_id = label_map[example.label]
        elif output_mode == "multi_classification":
            label_id = []
            for c in label_list:
                l = 1 if c in example.label else 0
                label_id.append(l)
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {} (id = {})".format(example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

def gpt2_convert_examples_to_features(examples, label_list, max_seq_length, seq_segments,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label : i for i, label in enumerate(label_list)}
    
    tokenizer.max_len = total_input_length = (max_seq_length * seq_segments)

    extra_tokens = (total_input_length // max_seq_length)
    tokenless_total_len = total_input_length - extra_tokens

    features = []
    for (ex_index, example) in enumerate(examples):
        
        if ex_index % 500 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            exit("cant handle b tokens right now")
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, total_input_length - 3)
        else:
            if len(tokens_a) > tokenless_total_len :
                tokens_a = tokens_a[:tokenless_total_len ]


        i = max_seq_length - 1
        while i < len(tokens_a):
            tokens_a.insert(i,"[CLS]")
            i+=max_seq_length

        tokens_a.append("[CLS]")
        input_ids = tokenizer.convert_tokens_to_ids(tokens_a)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.

        # Zero-pad up to the sequence length.
        padding = [0] * (total_input_length - len(input_ids))
        input_ids += padding

        assert len(input_ids) == total_input_length

        if output_mode == "classification":
            if example.label not in label_list:
                print("bad label on example {}, {}".format(ex_index,example.guid))
                continue
            label_id = label_map[example.label]
        elif output_mode == "multi_classification":
            label_id = []
            for c in label_list:
                l = 1 if c in example.label else 0
                label_id.append(l)
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens_a]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("label: {} (id = {})".format(example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=None,
                              segment_ids=None,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average='binary'):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

from collections import Counter
def precision_recall_f1(preds, y_true):
    precision, recall, fscore, support = score(y_true, preds)
    f1_micro  = f1_score(y_true=y_true, y_pred=preds, average="micro")
    f1_macro  = f1_score(y_true=y_true, y_pred=preds, average="macro")
    f1_weight = f1_score(y_true=y_true, y_pred=preds, average="weighted")

    acc = simple_accuracy(preds, y_true)
    f1 = f1_score(y_true=y_true, y_pred=preds, average=None)

    return {
        "acc": acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weight": f1_weight,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "support": support,
    }

def precision_recall_f1_multi(preds, y_true, labels):
    correct_labels = accuracy_score(y_true, preds)
    precision, recall, fscore, support = score(y_true, preds, labels=labels, average = 'micro')
    print("--GLOBAL Scores--")
    print('accuracy:  {}'.format(correct_labels))
    print('precision: {}'.format(precision))
    print('recall:    {}'.format(recall))
    print('fscore:    {}'.format(fscore))
    print('support:   {}'.format(support))

    print("--Class Scores--")

    precisions = []
    recalls = []
    fscores = []
    supports = []
    for i in labels:
        cur_y_true = y_true[:,i]
        cur_preds = preds[:,i]

        precision, recall, fscore, support = score(cur_y_true, cur_preds)
        precisions.append(precision[1])
        recalls.append(recall[1])
        fscores.append(fscore[1])
        supports.append(support)
    print('precision: {}'.format(precisions))
    print('recall:    {}'.format(recalls))
    print('fscore:    {}'.format(fscores))
    print('support:   {}'.format(supports))

    #print(Counter(preds))

    return {
        "prec": precision,
        "recall": recall,
    }



def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "uspto":
        return precision_recall_f1(preds, labels)
    elif task_name == "wireless":
        return precision_recall_f1(preds, labels)
    elif task_name == "agriculture":
        return precision_recall_f1(preds, labels)
    elif task_name == "chem":
        return precision_recall_f1(preds, labels)
    elif task_name == "computing":
        return precision_recall_f1(preds, labels)
    elif task_name == "subclass":
        return precision_recall_f1(preds, labels)
    elif task_name == "section":
        return precision_recall_f1(preds, labels)
    elif task_name == "arxiv":
        return precision_recall_f1(preds, labels)
    else:
        print("missing metric key")
        return precision_recall_f1(preds, labels)

processors = {
    "wireless": USPTOWirelessProcessor,
    "agriculture": USPTOAgProcessor,
    "chem" : USPTOChemProcessor,
    "computing" : USPTOComputingProcessor,
    "arxiv" : ArxivProcessor,
    "section": USPTOAllSectionProcessor,
    "subclass": USPTOAllSubClassProcessor,
}


output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "uspto": "classification",
    "wireless": "classification",
    "agriculture": "classification",
    "chem": "classification",
    "computing": "classification",
    "subclass": "classification",
    "section": "classification",
    "arxiv": "classification",
}
