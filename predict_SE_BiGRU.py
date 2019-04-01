

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
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling import BertForSentenceExtraction_BiGRU

import json
import random
from sklearn.metrics import precision_recall_fscore_support,recall_score,confusion_matrix
import re
from rouge import Rouge 

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

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
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MyDataProcessor(DataProcessor):

    def get_train_examples(self, data_dir, max_doc_sentence_len):
        self.max_doc_sentence_len = max_doc_sentence_len
        logger.info("LOADING TRAINING DATA : {}".format(data_dir))
        with open(data_dir,'r') as f:
            data = json.load(f)
        return self._create_examples(data)

    def get_dev_examples(self, data_dir, max_doc_sentence_len):
        self.max_doc_sentence_len = max_doc_sentence_len
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
        all_summary = []
        for idx,ele in tqdm(enumerate(data),desc='Loading Data'):

            doc_example = []
            doc_summary = []
            for sentence in ele['context']:

                if len(doc_example) < self.max_doc_sentence_len:

                    if sentence in ele['target']:
                        label = 1
                        doc_summary.append(sentence)
                    else:
                        label = 0
                    doc_example.append(self.InputExample(self.rm_punc(sentence),label))

            all_summary.append(self.rm_punc(" ".join(doc_summary)))
            all_examples.append(doc_example)

        sentence_len = [len(ele) for ele in all_examples]
        sort_order = np.argsort(sentence_len)[::-1]
        all_examples = [all_examples[i] for i in sort_order]
        all_summary = [all_summary[i] for i in sort_order]
        all_examples_len = [len(s) for s in all_examples]

        return all_examples, all_examples_len, all_summary






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
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument("--eval_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    ## Other parameters

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

    parser.add_argument("--model_dir_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Model path")

    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        required=True,
                        help="Model path")

    parser.add_argument("--start_epochs",
                        type=int,
                        required=True,
                        help="Use which epoch model")

    parser.add_argument("--end_epochs",
                        type=int,
                        required=True,
                        help="Use which epoch model")

    parser.add_argument("--output_result_path",
                        type=str,
                        required=True,
                        help="path of the output result")

    parser.add_argument("--max_doc_sentence_len",
                        default=32,
                        type=int,
                        help="Setting the max number of document of sentence")


    args = parser.parse_args()

    
    logger.info("Model Dir Path : {}".format(args.model_dir_path))
    logger.info("Model Name : {}".format(args.model_name))
    logger.info("Start Epochs : {}".format(args.start_epochs))
    logger.info("End Epochs : {}".format(args.end_epochs))
    logger.info("Output Result Path : {}".format(args.output_result_path))
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


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    processor = MyDataProcessor()
    num_labels = 2
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

        
    for ep in range(args.start_epochs,args.end_epochs + 1):
        eval_model_file = os.path.join(args.model_dir_path, "pytorch_model_{0}_epoch_{1}.bin".format(ep, args.model_name))

        logger.info("***** Only Do Evaluation *****")
        logger.info("Loading model : {}".format(eval_model_file))

        # Load a trained model that you have fine-tuned
        model_state_dict = torch.load(eval_model_file)
        model = BertForSentenceExtraction_BiGRU.from_pretrained(args.bert_model, state_dict=model_state_dict, num_labels=num_labels)
        
        model.to(device)
        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            eval_examples, eval_examples_doc_len, all_summary = processor.get_dev_examples(args.eval_data_dir, args.max_doc_sentence_len)
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
            rouge_1, rouge_2, rouge_L = 0, 0, 0
            eval_loss, eval_accuracy = 0, 0
            skip = 0
            eval_predict = []
            eval_true = []
            # print(all_doc_len)
            for index, batch in tqdm(enumerate(eval_dataloader), desc="Evaluating"):
                batch = tuple(t for t in batch)
                input_docs, input_doc_lens, input_doc_sentence_label = batch

                assert input_doc_lens[0].tolist() == all_doc_len[index].tolist()

                with torch.no_grad():
                    tmp_eval_loss = model(device, input_docs, input_doc_lens, input_doc_sentence_label)
                    logits, true_label = model(device, input_docs, input_doc_lens, input_doc_sentence_label,do_eval=True)
                
                logits = logits.detach().cpu().numpy()
                true_label = true_label.to('cpu').numpy().tolist()
                eval_true += true_label

                tmp_eval_accuracy = accuracy(logits, true_label)
                predict_label = np.argmax(logits, axis=1).tolist()
                eval_predict += predict_label

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy
                
                predict_sentence = [eval_examples[index][idx]['sentence'] for idx,i in enumerate(predict_label) if i == 1]
                true_sentence = [eval_examples[index][idx]['sentence'] for idx,i in enumerate(true_label) if i == 1]
                gold_summary = all_summary[index]

                try:
                    rouge_score = Rouge().get_scores(gold_summary," ".join(predict_sentence))
                    rouge_1 += rouge_score[0]['rouge-1']['f']
                    rouge_2 += rouge_score[0]['rouge-2']['f']
                    rouge_L += rouge_score[0]['rouge-l']['f']
                    
                except:
                    skip += 1
                
            final_rouge_1 = round(rouge_1/len(eval_examples),4)
            final_rouge_2 = round(rouge_2/len(eval_examples),4)
            final_rouge_L = round(rouge_L/len(eval_examples),4)

            skip_rouge_1 = round(rouge_1/(len(eval_examples)-skip),4) if len(eval_examples)-skip != 0 else 0
            skip_rouge_2 = round(rouge_2/(len(eval_examples)-skip),4) if len(eval_examples)-skip != 0 else 0
            skip_rouge_L = round(rouge_L/(len(eval_examples)-skip),4) if len(eval_examples)-skip != 0 else 0

            print('Using model :',eval_model_file)
            print(final_rouge_1, final_rouge_2, final_rouge_L)
            print(skip_rouge_1, skip_rouge_2, skip_rouge_L)
            print("output path :",args.output_result_path)
            # with open(args.output_result_path,'a') as f:
            #     f.write('Using model : {}\n'.format(eval_model_file))
            #     f.write('ROUGE_1 : {}, ROUGE_2 : {}, ROUGE_L : {}\n'.format(final_rouge_1,final_rouge_2,final_rouge_L))
            #     f.write('S_ROUGE_1 : {}, S_ROUGE_2 : {}, S_ROUGE_L : {}\n\n\n'.format(skip_rouge_1,skip_rouge_2,skip_rouge_L))

if __name__ == "__main__":
    main()
