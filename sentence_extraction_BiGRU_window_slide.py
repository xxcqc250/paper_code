'''
用BERT+RNN試試效果
BERT input :
[CLS] sentence [SEP]

BERT output :
取得[CLS] vector
再丟入GRU


'''

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
from modeling import BertForSentenceExtraction_BiGRU

import json
import random
from sklearn.metrics import precision_recall_fscore_support,recall_score,confusion_matrix
import re

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

        
# class DataProcessor(object):
#     """Base class for data converters for sequence classification data sets."""

#     def get_train_examples(self, data_dir):
#         """Gets a collection of `InputExample`s for the train set."""
#         raise NotImplementedError()

#     def get_dev_examples(self, data_dir):
#         """Gets a collection of `InputExample`s for the dev set."""
#         raise NotImplementedError()

#     def get_labels(self):
#         """Gets the list of labels for this data set."""
#         raise NotImplementedError()

#     @classmethod
#     def _read_tsv(cls, input_file, quotechar=None):
#         """Reads a tab separated value file."""
#         with open(input_file, "r", encoding='utf-8') as f:
#             reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
#             lines = []
#             for line in reader:
#                 lines.append(line)
#             return lines


class MyDataProcessor(object):
    def __init__(self, max_doc_sentence_len, window_size):
        self.max_doc_sentence_len = max_doc_sentence_len
        self.window_size = window_size

    def get_train_examples(self, data_dir):
        logger.info("LOADING TRAINING DATA : {}".format(data_dir))
        with open(data_dir,'r') as f:
            data = json.load(f)
        return self._create_examples(data)

    def get_dev_examples(self, data_dir):
        logger.info("LOADING EVALUATION DATA : {}".format(data_dir))
        with open(data_dir,'r') as f:
            data = json.load(f)
        return self._create_examples(data)

    def InputExample(self,sentence,label):
        tmp = {}
        tmp['sentence'] = sentence
        tmp['label'] = label
        return tmp

    def get_labels(self):
        return [0, 1]

    def rm_punc(self,text):
        return re.sub("[\s+`!#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+'", " ",text).strip() + '.'

    def get_pos(self,idx,window,context_size):
        start_pos = None
        end_pos = None  
        if idx - window < 0:
            start_pos = 0
        else:
            start_pos = idx - window

        if idx + window + 1 > context_size:
            end_pos = context_size
        else:
            end_pos = idx + window + 1
        return start_pos,end_pos

    def _create_examples(self,data):
        
        label = None
        all_examples = []
        for ele in tqdm(data,desc='Loading Data'):

            index = 0
            while True:
                s_pos, e_pos = self.get_pos(index, self.window_size, len(ele['context']))
                selected_sentence = ele['context'][s_pos:e_pos]

                doc_slide_example = []
                for sentence in selected_sentence:
                    if sentence in ele['target']:
                        label = 1
                    else:
                        label = 0
                    if len(doc_slide_example) < self.max_doc_sentence_len:
                        doc_slide_example.append(self.InputExample(self.rm_punc(sentence),label))

                all_examples.append(doc_slide_example)
                index += self.window_size
                if e_pos >= len(ele['context']):
                    break

        # sentence_len = [len(ele) for ele in all_examples]
        # sort_order = np.argsort(sentence_len)[::-1]
        # all_examples = [all_examples[i] for i in sort_order]
        all_examples_len = [len(s) for s in all_examples]

        return all_examples, all_examples_len





def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, max_doc_sentence_len):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    
    features = []
    logging_count = 0
    for doc in tqdm(examples,desc='Convert To Features'):

        doc_feature = []
        for example in doc:

            tokens_b = ""
            tokens_sentence = tokenizer.tokenize(example['sentence'])
            _truncate_seq_pair(tokens_sentence, tokens_b, max_seq_length - 2)

            tokens = ["[CLS]"] + tokens_sentence + ["[SEP]"]
            segment_ids = [0] * len(tokens)

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

            label_id = label_map[example['label']]
            if logging_count < 5:
                logging_count += 1
                logger.info("*** Example ***")
                logger.info("tokens: %s" % " ".join(
                        [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %d)" % (example['label'], label_id))

            doc_feature.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_id=label_id))


        padding_times = max_doc_sentence_len - len(doc_feature)
        while padding_times > 0:
            padding_times -= 1
            padding = [0] * max_seq_length
            doc_feature.append(
                    InputFeatures(input_ids=padding,
                                  input_mask=padding,
                                  segment_ids=padding,
                                  label_id=0))
        assert len(doc_feature) == max_doc_sentence_len
        features.append(doc_feature)
    return features

def InputFeatures(input_ids, input_mask, segment_ids, label_id):
    tmp = {}
    tmp['input_ids'] = input_ids
    tmp['input_mask'] = input_mask
    tmp['segment_ids'] = segment_ids
    tmp['label_id'] = label_id
    return tmp

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
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--window_size",
                        type=int,
                        default=2,
                        help="A paragraph's window size")

    ## Other parameters
    parser.add_argument("--eval_data_dir",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=5,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--gpu_device", 
                        type=str,
                        help="use gpu device number") 
    parser.add_argument("--output_model_name", 
                        type=str,
                        required=True,
                        help="Naming model and result") 
    parser.add_argument("--use_last_model",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--use_last_model_path",
                        default=None,
                        type=str,
                        help="Using last training model")

    parser.add_argument("--max_doc_sentence_len",
                        default=32,
                        type=int,
                        help="Setting the max number of document of sentence")
    parser.add_argument('--start_epoch',
                        type=int,
                        default=0,
                        help="start train epoch")

    parser.add_argument("--save_every_epoch",
                        action='store_true',
                        help="Saving models in every epoch")


    args = parser.parse_args()

    args.max_doc_sentence_len = args.window_size*2+1
    
    logger.info("save_every_epoch : {}".format(args.save_every_epoch))
    logger.info("Using Window Size : {}".format(args.window_size))
    logger.info("max_doc_sentence_len : {}".format(args.max_doc_sentence_len))
    logger.info("output_dir : {}".format(args.output_dir))
    logger.info("output_model_name : {}".format(args.output_model_name))
    logger.info("gpu_device : {}".format(args.gpu_device))
    logger.info("max_seq_length : {}".format(args.max_seq_length))
    logger.info("learning_rate : {}".format(args.learning_rate))
    logger.info("start_epoch : {}".format(args.start_epoch))
    logger.info("num_train_epochs : {}".format(args.num_train_epochs))
    logger.info("\n******************************************\n")
    
    if args.gpu_device != None:
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
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if args.do_train and "pytorch_model_{}.bin".format(args.output_model_name) in os.listdir(args.output_dir):
        raise ValueError("Output model name (pytorch_model_{}.bin) already exists.".format(args.output_model_name))
    os.makedirs(args.output_dir, exist_ok=True)

    



    processor = MyDataProcessor(args.max_doc_sentence_len, args.window_size)
    num_labels = 2
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None

    if args.do_train:

        try:

            logger.info("***** Try Loading train_examples & train_features*****")
            with open('DATA_CASHE/train_examples_BiGRU_window_slide_{0}_{1}_{2}'.format( args.data_dir.split('/')[-1], args.max_seq_length, args.max_doc_sentence_len),'r') as f:
                train_examples = json.load(f)
            with open('DATA_CASHE/train_examples_doc_len_BiGRU_window_slide_{0}_{1}_{2}'.format( args.data_dir.split('/')[-1], args.max_seq_length, args.max_doc_sentence_len),'r') as f:
                train_examples_doc_len = json.load(f)
            with open('DATA_CASHE/train_features_BiGRU_window_slide_{0}_{1}_{2}'.format(args.data_dir.split('/')[-1], args.max_seq_length, args.max_doc_sentence_len),'r') as f:
                train_features = json.load(f)

        except:

            logger.info("***** Create Training Data *****")
            train_examples, train_examples_doc_len = processor.get_train_examples(args.data_dir)
            with open('DATA_CASHE/train_examples_BiGRU_window_slide_{0}_{1}_{2}'.format(args.data_dir.split('/')[-1], args.max_seq_length, args.max_doc_sentence_len),'w') as f:
                json.dump(train_examples,f)
            with open('DATA_CASHE/train_examples_doc_len_BiGRU_window_slide_{0}_{1}_{2}'.format(args.data_dir.split('/')[-1], args.max_seq_length, args.max_doc_sentence_len),'w') as f:
                json.dump(train_examples_doc_len,f)
            train_features = convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer, args.max_doc_sentence_len)
            with open('DATA_CASHE/train_features_BiGRU_window_slide_{0}_{1}_{2}'.format(args.data_dir.split('/')[-1], args.max_seq_length, args.max_doc_sentence_len),'w') as f:
                json.dump(train_features,f)

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        all_doc_example = []
        all_doc_sentence_label = []
        for Doc_example in train_features:
            doc_tmp = []
            doc_sentence_label_tmp = []

            all_input_ids = [f['input_ids'] for f in Doc_example]
            all_input_mask = [f['input_mask'] for f in Doc_example]
            all_segment_ids = [f['segment_ids'] for f in Doc_example]
            all_label_ids = [f['label_id'] for f in Doc_example]
            doc_tmp.append(all_input_ids)
            doc_tmp.append(all_input_mask)
            doc_tmp.append(all_segment_ids)            
            all_doc_sentence_label.append(all_label_ids)

            all_doc_example.append(doc_tmp)

        all_doc_len = torch.tensor(train_examples_doc_len)
        all_doc_example = torch.tensor(all_doc_example)
        all_doc_sentence_label = torch.tensor(all_doc_sentence_label)

        train_data = TensorDataset(all_doc_example, all_doc_len, all_doc_sentence_label)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)


    

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    

    if args.do_train:
        # Prepare model
        if args.use_last_model:
            # 載入之前的model
            logger.info("Loading model : {}".format(args.use_last_model_path))
            training_modelpath = args.use_last_model_path
            model_state_dict = torch.load(training_modelpath)
            model = BertForSentenceExtraction_BiGRU.from_pretrained(args.bert_model, state_dict=model_state_dict) 
        else:
            logger.info("Loading Bert-base model...")
            model = BertForSentenceExtraction_BiGRU.from_pretrained(args.bert_model,
                  cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
                  num_labels = num_labels)


        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

        result = []
        model.train()
        for epoch in trange(int(args.start_epoch),int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            local_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                # batch = tuple(t.to(device) for t in batch)
                batch = tuple(t for t in batch)
                input_docs, input_doc_lens, input_doc_sentence_label = batch
                loss = 0
                loss = model(device, input_docs, input_doc_lens, input_doc_sentence_label)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                local_loss += loss.item()
                # nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if step % 20 == 0 and step != 0:
                    tmp = {'epoch': epoch, 'step': step, 'loss': local_loss/20.00}
                    result.append(tmp)
                    print(tmp)
                    local_loss = 0

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # 每次儲存都要重新載入參數
            if args.save_every_epoch:
                # Save a trained model
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.output_dir, "pytorch_model_{}_epoch_{}.bin".format(epoch,args.output_model_name))
                output_model_loss_file = os.path.join(args.output_dir, "model_loss_{}_epoch_{}.txt".format(epoch,args.output_model_name))
                logger.info("Saving model : {}".format(output_model_file))
                torch.save(model_to_save.state_dict(), output_model_file)
                json.dump(result, open(output_model_loss_file, "w"))
                model_to_save = None
                model = None

                if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                    # Load a trained model that you have fine-tuned
                    model_state_dict = torch.load(output_model_file)
                    model = BertForSentenceExtraction_BiGRU.from_pretrained(args.bert_model, state_dict=model_state_dict, num_labels=num_labels)
                    model.to(device)

                    eval_examples, eval_examples_doc_len = processor.get_dev_examples(args.eval_data_dir)
                    eval_features = convert_examples_to_features(
                        eval_examples, label_list, args.max_seq_length, tokenizer, args.max_doc_sentence_len)
                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    all_doc_example = []
                    all_doc_sentence_label = []
                    for Doc_example in eval_features:
                        doc_tmp = []
                        doc_sentence_label_tmp = []

                        all_input_ids = [f['input_ids'] for f in Doc_example]
                        all_input_mask = [f['input_mask'] for f in Doc_example]
                        all_segment_ids = [f['segment_ids'] for f in Doc_example]
                        all_label_ids = [f['label_id'] for f in Doc_example]
                        doc_tmp.append(all_input_ids)
                        doc_tmp.append(all_input_mask)
                        doc_tmp.append(all_segment_ids)            
                        all_doc_sentence_label.append(all_label_ids)

                        all_doc_example.append(doc_tmp)

                    all_doc_len = torch.tensor(eval_examples_doc_len)
                    all_doc_example = torch.tensor(all_doc_example)
                    all_doc_sentence_label = torch.tensor(all_doc_sentence_label)

                    eval_data = TensorDataset(all_doc_example, all_doc_len, all_doc_sentence_label)
                    # Run prediction for full data
                    eval_sampler = SequentialSampler(eval_data)
                    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                    model.eval()

                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0

                    eval_predict = []
                    eval_true = []
                    for batch in tqdm(eval_dataloader, desc="Evaluating"):
                        
                        batch = tuple(t for t in batch)
                        input_docs, input_doc_lens, input_doc_sentence_label = batch


                        with torch.no_grad():
                            tmp_eval_loss = model(device, input_docs, input_doc_lens, input_doc_sentence_label)
                            logits, true_label = model(device, input_docs, input_doc_lens, input_doc_sentence_label,do_eval=True)

                        logits = logits.detach().cpu().numpy()
                        true_label = true_label.to('cpu').numpy()
                        eval_true += true_label.tolist()

                        tmp_eval_accuracy = accuracy(logits, true_label)
                        eval_predict += np.argmax(logits, axis=1).tolist()

                        eval_loss += tmp_eval_loss.mean().item()
                        eval_accuracy += tmp_eval_accuracy

                        nb_eval_steps += 1

                    # print(eval_predict)
                    eval_loss = eval_loss / nb_eval_steps
                    loss = tr_loss/nb_tr_steps if args.do_train else None
                    result = {'eval_loss': eval_loss,
                              'global_step': global_step,
                              'loss': loss}

                    label_recall = recall_score(eval_true, eval_predict,average=None)
                    label_confusion_matrix = confusion_matrix(eval_true, eval_predict).ravel()
                    Pr,Re,F1,_ = precision_recall_fscore_support(eval_true, eval_predict, average='weighted')
                    output_eval_file = os.path.join(args.output_dir, "eval_results_{}_epoch_{}.txt".format(epoch,args.output_model_name))
                    with open(output_eval_file, "w") as writer:
                        writer.write("****** eval_results_{}.txt ******\n\n".format(args.output_model_name))
                        logger.info("***** Eval results *****")
                        logger.info("Precision : {}".format(Pr))
                        logger.info("Recall : {}".format(Re))
                        logger.info("F1 : {}".format(F1))
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n\n" % (key, str(result[key])))
                        writer.write("Precision : {}\nRecall : {}\nF1 : {}\n\n".format(Pr,Re,F1))
                        writer.write("label 0 Recall : {}\nlabel 1 Recall : {}\n\n".format(label_recall[0],label_recall[1]))
                        writer.write("tn : {}  把0判給0\n".format(label_confusion_matrix[0]))
                        writer.write("fp : {}  把0判給1\n".format(label_confusion_matrix[1]))
                        writer.write("fn : {}  把1判給0\n".format(label_confusion_matrix[2]))
                        writer.write("tp : {}  把1判給1".format(label_confusion_matrix[3]))

                # Prepare model
                # 載入之前的model
                model = None
                logger.info("Loading model : {}".format(output_model_file))
                training_modelpath = output_model_file
                model_state_dict = torch.load(training_modelpath)
                model = BertForSentenceExtraction_BiGRU.from_pretrained(args.bert_model, state_dict=model_state_dict) 

                model.to(device)
                if n_gpu > 1:
                    model = torch.nn.DataParallel(model)

                # Prepare optimizer
                param_optimizer = list(model.named_parameters())
                no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                    ]
                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=args.learning_rate,
                                     warmup=args.warmup_proportion,
                                     t_total=num_train_optimization_steps)

                result = []
                model.train()



if __name__ == "__main__":
    main()
