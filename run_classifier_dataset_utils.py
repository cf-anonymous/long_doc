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

    def __init__(self, input_ids, input_mask, segment_ids, label_id, guid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.guid = guid


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
    def _read_tsv(cls, input_file, quotechar=None, encoding='utf-8'):
        """Reads a tab separated value file."""
        csv.field_size_limit(sys.maxsize)
        
        with open(input_file, "r", encoding=encoding) as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), 
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class USPTOProcessor(DataProcessor):
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
        return ["A","B","C","D","E","F","G","H"]

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


class USPTOAllSubProcessor(DataProcessor):
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
        return ['G11B', 'A41D', 'A63B', 'E03D', 'E03C', 'A47C', 'A47G', 'A47B', 'B26B', 'A46B', 'A47K', 'E05F', 'E05D', 'D01G', 'B43K', 'A44B', 'H01L', 'H05B', 'B65B', 'F01B', 'D04B', 'G01F', 'F16H', 'B65H', 'A01J', 'F02B', 'A61F', 'F16K', 'A47H', 'B25F', 'B66B', 'B25B', 'E01D', 'B08B', 'B42F', 'A41F', 'B21C', 'B23P', 'B23Q', 'B21D', 'H01S', 'H02K', 'H01F', 'H01R', 'H05K', 'H01B', 'A44C', 'G01B', 'B43L', 'F26B', 'A43B', 'A43C', 'E02F', 'E01H', 'B42D', 'G09F', 'F41A', 'F41G', 'A01K', 'F16G', 'A01M', 'A01G', 'E06B', 'E04B', 'E04F', 'B68C', 'A01D', 'D02G', 'F02C', 'F01N', 'F16D', 'B60T', 'F24H', 'F01K', 'F02K', 'F25B', 'F25C', 'B60H', 'F25D', 'A63C', 'A47F', 'F28D', 'C03B', 'E05B', 'B60R', 'B21B', 'B21J', 'G01P', 'G01N', 'G01M', 'G02B', 'G01L', 'E01C', 'B60C', 'G01G', 'G05G', 'B60D', 'F16C', 'B25G', 'B23B', 'B23D', 'B26D', 'F42B', 'F41F', 'F15B', 'A47J', 'B41F', 'B41M', 'B41N', 'B41L', 'F23B', 'F23G', 'A01C', 'B63B', 'B63G', 'C23C', 'F22B', 'F01P', 'F02F', 'F01L', 'F02M', 'F02N', 'F02D', 'F02P', 'F28F', 'F41B', 'B28D', 'C10L', 'E04D', 'A61M', 'B63C', 'A62B', 'A61B', 'A24B', 'A41G', 'G05D', 'B32B', 'F16L', 'D03C', 'D03D', 'B21F', 'B65D', 'B27M', 'B60B', 'B22D', 'B22C', 'E21B', 'C04B', 'A62C', 'A01B', 'B25D', 'B62D', 'B60K', 'B60V', 'G01V', 'E04G', 'G04G', 'F01M', 'B62B', 'B61H', 'F16F', 'H02G', 'F04B', 'B60L', 'F17C', 'B23K', 'G06K', 'B65G', 'G07F', 'A22C', 'B01D', 'B66C', 'A61J', 'A24F', 'B67D', 'A45F', 'B25C', 'B26F', 'G06F', 'G07C', 'E01B', 'B05B', 'B02C', 'G03B', 'B64C', 'F16M', 'B66D', 'E04H', 'B01F', 'A63F', 'F01D', 'F16J', 'B25H', 'B62J', 'B62H', 'A61G', 'E05C', 'B25J', 'B60N', 'B60J', 'B60P', 'A47D', 'E21C', 'B60Q', 'B41J', 'G02C', 'F21V', 'E01F', 'B60G', 'E02B', 'A45C', 'H02N', 'G02F', 'H04B', 'C21C', 'A61K', 'G01R', 'H04L', 'F21L', 'F21S', 'B64D', 'B28C', 'B29B', 'G01J', 'G01K', 'G01D', 'G03D', 'B05C', 'E21D', 'E02D', 'F16B', 'B42B', 'F03B', 'F04D', 'F03D', 'F04C', 'D01D', 'B29C', 'F23N', 'C08J', 'F27D', 'A61D', 'A61C', 'G09B', 'A63H', 'H02B', 'B63H', 'H01J', 'A41C', 'B24B', 'B24D', 'A21C', 'B04B', 'A61N', 'H04R', 'A61H', 'B31B', 'C30B', 'C07D', 'D06P', 'H01G', 'H01M', 'B01J', 'C22C', 'B22F', 'C21B', 'B05D', 'C09D', 'C08L', 'C09C', 'C09B', 'C06B', 'B28B', 'B65C', 'D21G', 'C10B', 'C25D', 'C25F', 'C25B', 'C02F', 'C25C', 'C10C', 'C10G', 'E03F', 'B44C', 'C03C', 'C09K', 'C07C', 'G09G', 'C08F', 'C01B', 'C01D', 'C01C', 'C01G', 'C01F', 'A01N', 'C12N', 'C07H', 'C07K', 'C12P', 'C07J', 'A23G', 'A21D', 'A23C', 'A23L', 'A23N', 'C08G', 'H05H', 'F23C', 'A23J', 'G03F', 'C12Q', 'G03G', 'G03C', 'C11D', 'C12M', 'A47L', 'A61P', 'C08K', 'C08C', 'A01H', 'C07B', 'C07F', 'G21F', 'G10F', 'G10D', 'G10H', 'H01H', 'A21B', 'F27B', 'H04N', 'G08B', 'G01T', 'G21K', 'C23F', 'H01K', 'H02J', 'H03H', 'G05B', 'G05F', 'H02P', 'G01S', 'H03K', 'H03L', 'H03F', 'H03B', 'G06G', 'H03G', 'H01P', 'H01C', 'H04Q', 'H04M', 'G06E', 'G08G', 'H03M', 'H01Q', 'G06T', 'H04H', 'G03H', 'F16P', 'G01C', 'B41B', 'H02H', 'H02M', 'G11C', 'G04C', 'G04B', 'H04J', 'G08C', 'H04K', 'H03D', 'H05G', 'G01W', 'G04F', 'G10L', 'G06Q', 'G07B', 'G06N', 'G06C', 'B60S', 'A42B', 'F41C', 'E04C', 'F03C', 'E03B', 'D06C', 'D06F', 'B67B', 'A23F', 'B41K', 'B41C', 'F23K', 'F23J', 'D05B', 'B65F', 'A45D', 'E06C', 'B61G', 'B27F', 'G07G', 'F24D', 'F23D', 'B02B', 'B64G', 'B62M', 'B62K', 'F21K', 'B43M', 'F28G', 'B66F', 'B61D', 'B42C', 'F01C', 'B29D', 'F23M', 'B24C', 'G07D', 'F24F', 'A63J', 'D01F', 'B31F', 'D06M', 'C23G', 'C21D', 'A61L', 'D04H', 'G06M', 'G21C', 'H03J', 'H05F', 'G09C', 'D05C', 'B30B', 'D01H', 'D02J', 'B21K', 'A43D', 'B68B', 'F23R', 'F02G', 'D03J', 'F23L', 'G10K', 'B60M', 'B07C', 'B07B', 'B04C', 'B64B', 'F41J', 'B23C', 'A01F', 'C22F', 'C09J', 'D21H', 'D21F', 'B09C', 'D06B', 'A23D', 'G12B', 'C08B', 'A62D', 'B23H', 'B01B', 'B61C', 'H01T', 'G21G', 'A41B', 'F41H', 'B68G', 'B03B', 'F24J', 'F25J', 'B27B', 'B27C', 'F16N', 'C10J', 'F17D', 'A63D', 'B03C', 'D21C', 'D21D', 'A23B', 'C12H', 'C05G', 'B82B', 'B06B', 'H03C', 'G09D', 'B44F', 'B21H', 'F23Q', 'A45B', 'B03D', 'G10G', 'A46D', 'A23P', 'A22B', 'A63G', 'C22B', 'D21B', 'B01L', 'C10M', 'A42C', 'F04F', 'B27L', 'D07B', 'F15D', 'A41H', 'B62L', 'E21F', 'F15C', 'B09B', 'B23F', 'A63K', 'C14C', 'C09H', 'G10C', 'G06J', 'F03H', 'B61F', 'F16T', 'B61K', 'B64F', 'F21Y', 'D06L', 'B27D', 'B27N', 'B61L', 'C05F', 'G01H', 'G06D', 'D04C', 'B63J', 'F24C', 'B27G', 'B67C', 'B62C', 'F21W', 'C09G', 'B61B', 'A23K', 'C12C', 'C09F', 'C11C', 'F03G', 'E02C', 'F42C', 'F24B', 'B60F', 'B21G', 'B31C', 'C12J', 'F22D', 'F28B', 'B27K', 'C08H', 'A01L', 'H05C', 'B23G', 'D06Q', 'D02H', 'A24D', 'B60W', 'D01C', 'C05C', 'F42D', 'B27J', 'B81B', 'B41D', 'A61Q', 'E05G', 'B81C', 'G21B', 'A24C', 'G21D', 'D21J', 'C12S', 'C11B', 'B21L', 'H04S', 'C06C', 'B61J', 'C10F', 'B44B', 'C07G', 'B31D', 'B44D', 'A01P', 'C05D', 'D01B', 'F28C', 'B27H', 'C06F', 'C12R', 'D04D', 'F22G', 'G04D', 'C12G', 'F16S', 'D06H', 'A99Z', 'D06N', 'C06D', 'C40B', 'G10B', 'G21H', 'C13K', 'C05B', 'F23H', 'B29L', 'F21H', 'F17B', 'C14B', 'C10K', 'D04G', 'C12F', 'H04W', 'C10H', 'C23D', 'D06G', 'G01Q', 'D06J', 'B41G', 'B82Y', 'C13B', 'C10N', 'B29K']

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

from concurrent import futures
def map_fn(fn, *iterables):
    with futures.ProcessPoolExecutor(max_workers=16) as executor:
        result_iterator = executor.map(fn, *iterables)

    return [i for i in result_iterator]

import itertools

def parallel_convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    label_map = {label : i for i, label in enumerate(label_list)}
    

    results =  map_fn(run_one_step, examples, range(len(examples)), itertools.repeat(max_seq_length), itertools.repeat(tokenizer), itertools.repeat(output_mode), itertools.repeat(label_map))
    return results 


def run_one_step(example, ex_index, max_seq_length, tokenizer, output_mode, label_map):
    if ex_index % 1000 == 0:
        logger.info("Writing example %d" % (ex_index))

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]


    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if output_mode == "classification":
        label_id = label_map[example.label]
    elif output_mode == "regression":
        label_id = float(example.label)
    else:
        raise KeyError(output_mode)


    
    return InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          guid=example.guid)







def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}



    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

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
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
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
                              label_id=label_id,
                              guid=example.guid))
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
    elif task_name == "chem":
        return precision_recall_f1(preds, labels)
    elif task_name == "computing":
        return precision_recall_f1(preds, labels)
    elif task_name == "uspto_allsub":
        return precision_recall_f1(preds, labels)
    elif task_name == "arxiv":
        return precision_recall_f1(preds, labels)
    else:
        print("missing metric key")
        return precision_recall_f1(preds, labels)

processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "uspto": USPTOProcessor,
    "wireless": USPTOWirelessProcessor,
    "computing" : USPTOComputingProcessor,
    "uspto_allsub": USPTOAllSubProcessor,
    "chem" : USPTOChemProcessor,
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
    "computing": "classification",
    "chem": "classification",
    "uspto_allsub": "classification",
    "subclass": "classification",
    "section": "classification",
    "arxiv": "classification",
}
