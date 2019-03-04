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
"""Run BERT on SQuAD."""


################## [CLS] context [SEP] qusetion [SEP]  =＞ [CLS] qusetion [SEP] ##################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import math
import os
import random
import pickle
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from modeling import BertForGenerative
from optimization import BertAdam
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class SquadData(object):
    """A single training/test example for the Squad dataset."""

    def __init__(self,
                 question_text,
                 doc_tokens,
                 target_text=""):
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.target_text = target_text

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 output_ids):
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.output_ids = output_ids

def read_squad_data(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    datas = []
    for entry in tqdm(input_data,desc='read_squad_data'):

        doc_tokens_list = []
        for context_list in entry['context']:
            # doc_tokens 就是 context_token
            # context : The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France.
            # 使用空白、換行斷詞  ex:['The', 'Normans', '(Norman:', 'Nourmands;', 'French:', 'Normands;', 'Latin:', 'Normanni)', 'were', 'the', 'people', 'who', 'in', 'the', '10th', 'and', '11th', 'centuries', 'gave', 'their', 'name', 'to', 'Normandy,', 'a', 'region', 'in', 'France.']
            doc_tokens = []
            prev_is_whitespace = True
            for c in context_list:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
            doc_tokens_list.append(doc_tokens)
        
        # question_text = entry["question"]
        question_text = []
        target_text = entry["target"]
        data = SquadData(
            question_text=question_text,
            doc_tokens=doc_tokens_list,
            target_text=target_text)
        datas.append(data)
    
    return datas


def convert_data_to_features(data, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""
    continue_count = 0

    features = []
    for (index, ele) in tqdm(enumerate(data),desc='convert_data_to_features'):

        ## Start 先將文章太長的過濾掉 -------

        if len(ele.doc_tokens) > 30:
            continue_count += 1
            continue

        ## End ----------------------------


        ## Start 先拆開算最大長度，最後再合併 --------------
        ## 也會把output太長的給去除
        target_text_tokenize = [tokenizer.tokenize(d) for d in ele.target_text]

        Max_query_length = max_query_length - len(ele.target_text) - 1
        target_text = []
        target_len = 0
        continue_flag = False
        for text_list in target_text_tokenize:
            if target_len + len(text_list) < Max_query_length:
                target_text.append(text_list)
                target_len += len(text_list)
            else:
                continue_flag = True

        if continue_flag == True:
            continue_count += 1
            continue


        
        
        ## End ------------------------------------------




        all_doc_tokens = []                 # 經過model的tokenize斷詞(包含屬性) ['the', 'norman', '##s', '(', 'norman', ':', 'no', '##ur', '##man', '##ds', ';', 'french', ':', 'norman', '##ds', ';', 'latin', ':', 'norman', '##ni', ')', 'were', 'the', 'people', 'who', 'in', 'the', '10th', 'and', '11th', 'centuries', 'gave', 'their', 'name', 'to', 'normandy', ',', 'a', 'region', 'in', 'france', '.']
        tok_to_orig_index = []              # 經過tokenize後token的index [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 22, 23, 24, 25, 26, 26]
        orig_to_tok_index = []              # 經過tokenize後token起始位置 [0, 1, 3, 6, 11, 13, 16, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40]
        count = 0
        tokens_Len = 0
        cut_point = []

        for (i, tokens) in enumerate(ele.doc_tokens):
            for (k,token) in enumerate(tokens):
                orig_to_tok_index.append(tokens_Len + k)
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(count)
                    all_doc_tokens.append(sub_token)
                count += 1

            tokens_Len = len(all_doc_tokens)
            cut_point.append(tokens_Len)
        
        # The -3 accounts for [CLS], [SEP] and [SEP]
        # 我的會有很多個[SEP]也要扣除
        # 所以我應該要先max_seq_length - 1(先扣掉[CLS])，在看如果有 k 段句子再扣掉 k
        # max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        max_tokens_for_doc = max_seq_length - 1 - len(ele.doc_tokens)

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        # 這邊是說有些seq會超過max_seq_length，超過的話，會幫我們切成一段一段
        # 假設max_seq_length = 10, 而我們的input長度13，會切成(0,10),(10,13)這兩段
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)
        
        # 開始建立input格式([CLS]...[SEP]...[SEP]...)
        for (doc_span_index, doc_span) in enumerate(doc_spans):

            tokens = []
            output_tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            # 處理input地方
            seg_id_num = 0
            tokens.append("[CLS]")
            segment_ids.append(seg_id_num)
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                if split_token_index in cut_point:
                    tokens.append("[SEP]")
                    segment_ids.append(seg_id_num)
                    seg_id_num += 1
                else:
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                    is_max_context = _check_is_max_context(doc_spans, doc_span_index,split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context

                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(seg_id_num)

            tokens.append("[SEP]")
            segment_ids.append(seg_id_num)
            seg_id_num += 1


            # 處理output地方
            output_tokens.append("[CLS]")
            for token_list in target_text:
                for token in token_list:
                    output_tokens.append(token)            
                output_tokens.append("[SEP]")

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            while len(output_ids) < max_seq_length:    
                output_ids.append(-1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(output_ids) == max_seq_length

            # logger.info("*** Example ***")
            # logger.info("doc_span_index: %s" % (doc_span_index))
            # logger.info("tokens: %s" % " ".join(tokens))
            # logger.info("token_to_orig_map: %s" % " ".join([
            #     "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
            # logger.info("token_is_max_context: %s" % " ".join([
            #     "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
            # ]))
            # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            # logger.info(
            #     "input_mask: %s" % " ".join([str(x) for x in input_mask]))
            # logger.info(
            #     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("output_ids: %s" % " ".join([str(x) for x in output_ids]))
            # logger.info("output_tokens: %s" % " ".join(output_tokens))
            features.append(
                InputFeatures(
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    output_ids=output_ids
                    ))
    print('skip num :',continue_count)
    return features

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    print('outputs',outputs)
    print('labels',labels)
    return np.sum(outputs == labels)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--train_file", default=None, type=str, help="json for training.")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="json for predictions.")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - b est_non_null is greater than the threshold predict null.")
    parser.add_argument("--gpu_device", default=0, type=str, required=True,
                        help="use gpu device number")    

    args = parser.parse_args()

    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory () already exists and is not empty.")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    # tokenizer.max_len=800

    train_datas = None
    num_train_steps = None


    if args.do_train:
        try:
            with open(args.train_file + "_squad_data", "rb") as reader:
                train_datas = pickle.load(reader)
            print("Loading Squad data")
        except:
            train_datas = read_squad_data(
                input_file=args.train_file, is_training=True)
            with open(args.train_file + "_squad_data", "wb") as writer:
                pickle.dump(train_datas, writer)

        num_train_steps = int(
            len(train_datas) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model

    # 載入之前的model
    training_modelpath = '/notebooks/github/pytorch-pretrained-BERT/examples/result_dir_all_label/using_abstract_dataset/20_epoch/pytorch_model.bin'
    model_state_dict = torch.load(training_modelpath)
    print('model load OK')
    model = BertForGenerative.from_pretrained(args.bert_model, state_dict=model_state_dict)

    # # 使用pre-train model
    # print('Using original BERT model....')
    # model = BertForGenerative.from_pretrained(args.bert_model,
    #             cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    # elif n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

   
    global_step = 0
    if args.do_train:
        cached_train_features_file = args.train_file+'_{0}_{1}_{2}_{3}'.format(
            args.bert_model, str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))
        train_features = None
        try:
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        except:
            train_features = convert_data_to_features(
                data=train_datas,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True)
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving train features into cached file %s", cached_train_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)
        logger.info("***** Running training *****")
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_output_ids = torch.tensor([f.output_ids for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_output_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        result = []
        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in  enumerate(tqdm(train_dataloader, desc="Iteration")):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, output_ids = batch
                loss = model(input_ids, segment_ids, input_mask, output_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                if step % 100 == 0:
                    # logits = model(input_ids, segment_ids, input_mask)
                    # logits = logits.detach().cpu().numpy()
                    # output_ids = output_ids.to('cpu').numpy()
                    # tmp_accuracy = accuracy(logits, output_ids)
                    # tmp = {'epoch': epoch, 'step': step, 'loss': loss.item(), 'accuracy' : tmp_accuracy}
                    tmp = {'epoch': epoch, 'step': step, 'loss': loss.item()}
                    result.append(tmp)
                    print(tmp)
                

    # Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

    output_model_loss_file = os.path.join(args.output_dir, "model_loss.txt")
    json.dump(result, open(output_model_loss_file, "w"))

if __name__ == "__main__":
    main()
