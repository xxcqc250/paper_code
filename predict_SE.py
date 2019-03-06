# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
import pickle
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling import BertForSentenceExtraction_DeepHidden,BertForSequenceClassification

import json
import random
from sklearn.metrics import recall_score
import re

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    def __init__(self,doc_text,target_text,original_text,label=None):
        self.target_text = target_text
        self.doc_text = doc_text
        self.label = label
        self.original_text = original_text
        

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    # @classmethod
    # def _read_tsv(cls, input_file, quotechar=None):
    #     """Reads a tab separated value file."""
    #     with open(input_file, "r", encoding='utf-8') as f:
    #         reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
    #         lines = []
    #         for line in reader:
    #             lines.append(line)
    #         return lines


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class MyDataProcessor(DataProcessor):

    def get_test_examples(self, data_dir):
        logger.info("LOADING TESTING DATA : {}".format(data_dir))
        with open(data_dir,'r') as f:
            data = f.read().replace('\n',' ')
        data = data.split('. ')

        return self._create_examples(data)



    def _create_examples(self,data,window=2):
        
        label = None
        all_examples = []

        context_index = [i for i in range(len(data))]
        for idx in context_index:
            start_pos = None
            end_pos = None
            start_pos, end_pos = self.get_pos(idx,window,len(data))

            target_text = self.rm_punc(data[idx])
            original_text = data[idx]
            doc_text = self.rm_punc(" ".join([data[n] for n in range(start_pos,end_pos)]))
            print(target_text)
            print(original_text)
            print(doc_text)
            exit()

            all_examples.append(InputExample(doc_text,target_text,original_text))

        return all_examples

    def get_labels(self):
        return [0, 1]

    def rm_punc(self,text):
        return re.sub("[\s+`!#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+'", " ",text).strip()

    def get_pos(self,idx,window,context_size):
        start_pos = None
        end_pos = None  
        if idx - window < 0:
            start_pos = 0
        else:
            start_pos = idx - window

        if idx + window > context_size:
            end_pos = context_size
        else:
            end_pos = idx + window
        return start_pos,end_pos


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    
    print(label_map)
    features = []
    logging_count = 0
    for example in tqdm(examples,desc='Convert To Features'):
        tokens_doc = tokenizer.tokenize(example.doc_text)
        tokens_sentence = tokenizer.tokenize(example.target_text)
        _truncate_seq_pair(tokens_sentence, tokens_doc, max_seq_length - 3)

        tokens = ["[CLS]"] + tokens_sentence + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_doc:
            tokens += tokens_doc + ["[SEP]"]
            segment_ids += [1] * (len(tokens_doc) + 1)

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

        label_id = label_map[example.label]
        if logging_count < 5:
            logging_count += 1
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
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

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--gpu_device", 
                        type=str,
                        required=True,
                        help="use gpu device number") 
    

    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    processor = MyDataProcessor()
    num_labels = 2

    examples = processor.get_test_examples(args.data_dir)


    

if __name__ == "__main__":
    main()
