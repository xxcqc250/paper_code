###
# 可自動儲存每個epoch的model
# 自動eval每個epoch的model
# 平衡label數量，使用harry方式平衡 :
# 
# 一篇文章，如果key sentence有n句，找另外n句非key sentence當作input
# 
###

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
from modeling import BertForSentenceExtraction_DeepHidden,BertForSequenceClassification,BertForSentenceExtraction_CNN

import json
import random
from sklearn.metrics import precision_recall_fscore_support,recall_score,confusion_matrix
import re
from rouge import Rouge 

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    def __init__(self,doc_text,target_text,summ_text=None):
        self.target_text = target_text
        self.doc_text = doc_text
        self.summ_text = summ_text
        self.label = 0
        

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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class MyDataProcessor(DataProcessor):

    def get_dev_examples(self, data_dir, window_size=2):
        logger.info("LOADING EVALUATION DATA : {}".format(data_dir))
        with open(data_dir,'r') as f:
            data = json.load(f)
        return self._create_eval_examples(data,window_size)

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

        if idx + window > context_size:
            end_pos = context_size
        else:
            end_pos = idx + window
        return start_pos,end_pos+1

    def _create_eval_examples(self,data,window=2):
        all_examples = []
        for idx,ele in enumerate(data):
            doc_examples = []
            context_index = [i for i in range(len(ele['context']))]

            for idx in context_index:
                start_pos = None
                end_pos = None
                start_pos, end_pos = self.get_pos(idx,window,len(ele['context']))
                
                # original_text = ele['context'][idx]

                select_sentence = ele['context'][start_pos:end_pos]
                doc_text = " ".join([self.rm_punc(s) for s in select_sentence])
                for sentence in select_sentence:
                    target_text = self.rm_punc(sentence)
                    summ_text = self.rm_punc(ele['summary'])
                    doc_examples.append(InputExample(doc_text,target_text,summ_text))
                    # print(doc_text)
                    # print(target_text)
                    # print(summ_text)
                    # input()
            all_examples.append(doc_examples)

        return all_examples





def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    
    # print(label_map)
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
        if logging_count < -1:
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
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")

    # parser.add_argument("--output_dir",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--window_size",
                        required=True,
                        type=int,
                        help="A paragraph's window size")

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

    parser.add_argument("--use_last_model",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--use_last_model_path",
                        default=None,
                        type=str,
                        help="Using last training model")

    parser.add_argument("--save_every_epoch",
                        action='store_true',
                        help="Saving models in every epoch")

    parser.add_argument("--output_result_path",
                        type=str,
                        required=True,
                        help="path of the output result")


    args = parser.parse_args()

    
    logger.info("save_every_epoch : {}".format(args.save_every_epoch))
    logger.info("Using Window Size : {}".format(args.window_size))
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


    eval_model_file = args.use_last_model_path
    logger.info("***** Only Do Evaluation *****")
    logger.info("Loading model : {}".format(eval_model_file))

    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(eval_model_file)
    # model = BertForSentenceExtraction_DeepHidden.from_pretrained(args.bert_model, state_dict=model_state_dict, num_labels=num_labels)
    model = BertForSentenceExtraction_CNN.from_pretrained(args.bert_model, 
                    state_dict=model_state_dict,
                    num_labels = num_labels,
                    sequence_length=args.max_seq_length) 
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.eval_data_dir, window_size=args.window_size)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        rouge_1, rouge_2, rouge_L = 0, 0, 0
        skip = 0
        for doc_examples in tqdm(eval_examples,desc='Examples'):
            # print(len(doc_examples))
            gold_summary = doc_examples[0].summ_text

            eval_features = convert_examples_to_features(
                doc_examples, label_list, args.max_seq_length, tokenizer)
        
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            # eval_loss, eval_accuracy = 0, 0
            # nb_eval_steps, nb_eval_examples = 0, 0

            # eval_predict = []
            # eval_true = []
            predict_label = []

            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                # eval_true += label_ids.tolist()

                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                # label_ids = label_ids.to(device)


                with torch.no_grad():
                    # tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                    logits = model(input_ids, segment_ids, input_mask)

                logits = logits.detach().cpu().numpy()
                # label_ids = label_ids.to('cpu').numpy()

                # tmp_eval_accuracy = accuracy(logits, label_ids)
                predict_label += np.argmax(logits, axis=1).tolist()
            # print(len(predict_label))

            extract_sentence = [doc_examples[idx].target_text for idx,p_label in enumerate(predict_label) if p_label==1]

            temp = extract_sentence
            extract_sentence = []
            for s in temp:
                if s not in extract_sentence:
                    extract_sentence.append(s)

            # extract_sentence = list(set(extract_sentence))
            # print(list(set(extract_sentence)))
            try:
                rouge_score = Rouge().get_scores(gold_summary," ".join(extract_sentence))
                rouge_1 += rouge_score[0]['rouge-1']['f']
                rouge_2 += rouge_score[0]['rouge-2']['f']
                rouge_L += rouge_score[0]['rouge-l']['f']
            except:
                skip += 1
            # rouge_2 = Rouge().get_scores(gold_summary," ".join(extract_sentence))
            # rouge_L = Rouge().get_scores(gold_summary," ".join(extract_sentence))
        
        final_rouge_1 = round(rouge_1/len(eval_examples),4)
        final_rouge_2 = round(rouge_2/len(eval_examples),4)
        final_rouge_L = round(rouge_L/len(eval_examples),4)

        skip_rouge_1 = round(rouge_1/(len(eval_examples)-skip),4)
        skip_rouge_2 = round(rouge_2/(len(eval_examples)-skip),4)
        skip_rouge_L = round(rouge_L/(len(eval_examples)-skip),4)

        print('Using model :',eval_model_file)
        print(final_rouge_1, final_rouge_2, final_rouge_L)
        print(skip_rouge_1, skip_rouge_2, skip_rouge_L)
        print("output path :",args.output_result_path)
        with open(args.output_result_path,'a') as f:
            f.write('Using model : {}\n'.format(eval_model_file))
            f.write('ROUGE_1 : {}, ROUGE_2 : {}, ROUGE_L : {}\n'.format(final_rouge_1,final_rouge_2,final_rouge_L))
            f.write('S_ROUGE_1 : {}, S_ROUGE_2 : {}, S_ROUGE_L : {}\n\n\n'.format(skip_rouge_1,skip_rouge_2,skip_rouge_L))
        exit()

                # eval_loss += tmp_eval_loss.mean().item()
                # eval_accuracy += tmp_eval_accuracy

                # nb_eval_examples += input_ids.size(0)
                # nb_eval_steps += 1

            # print(eval_predict)
            # eval_loss = eval_loss / nb_eval_steps
            # eval_accuracy = eval_accuracy / nb_eval_examples
            # loss = tr_loss/nb_tr_steps if args.do_train else None
            # result = {'eval_loss': eval_loss,
            #           'eval_accuracy': eval_accuracy,
            #           'global_step': global_step,
            #           'loss': loss}

        # label_recall = recall_score(eval_true, eval_predict,average=None)
        # label_confusion_matrix = confusion_matrix(eval_true, eval_predict).ravel()
        # Pr,Re,F1,_ = precision_recall_fscore_support(eval_true, eval_predict, average='weighted')
        # output_eval_file = os.path.join(args.output_dir, "eval_results_{}.txt".format(args.output_model_name))
        # with open(output_eval_file, "w") as writer:
        #     writer.write("****** eval_results_{}.txt ******\n\n".format(args.output_model_name))
        #     logger.info("***** Eval results *****")
        #     logger.info("Precision : {}".format(Pr))
        #     logger.info("Recall : {}".format(Re))
        #     logger.info("F1 : {}".format(F1))
        #     for key in sorted(result.keys()):
        #         logger.info("  %s = %s", key, str(result[key]))
        #         writer.write("%s = %s\n\n" % (key, str(result[key])))
        #     writer.write("Precision : {}\nRecall : {}\nF1 : {}\n\n".format(Pr,Re,F1))
        #     writer.write("label 0 Recall : {}\nlabel 1 Recall : {}\n\n".format(label_recall[0],label_recall[1]))
        #     writer.write("tn : {}  把0判給0\n".format(label_confusion_matrix[0]))
        #     writer.write("fp : {}  把0判給1\n".format(label_confusion_matrix[1]))
        #     writer.write("fn : {}  把1判給0\n".format(label_confusion_matrix[2]))
        #     writer.write("tp : {}  把1判給1".format(label_confusion_matrix[3]))


if __name__ == "__main__":
    main()
