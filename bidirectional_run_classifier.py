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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import random

from os.path import isfile

from tqdm import tqdm, trange

import numpy as np

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, Sigmoid

from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from run_classifier_dataset_utils_lstm import processors, output_modes, convert_examples_to_features, compute_metrics
from model import  NormalBert, LongBert, AblationLongBert, AttentionLongBert

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


logger = logging.getLogger(__name__)

from collections import Counter
def calculate_class_weights(labels):
    ret = Counter(labels)

    import pdb; pdb.set_trace()

    return ret


def get_model(args, num_labels):
    exp_type = args.experiment

    if exp_type == "attention":
        model = AttentionLongBert.from_pretrained(args.bert_model, args.seq_segments, args.max_seq_length, num_labels = num_labels)
    elif exp_type == "base":
        model = NormalBert.from_pretrained(args.bert_model, num_labels = num_labels)
    elif exp_type == "long":
        model = LongBert.from_pretrained(args.bert_model, args.seq_segments, args.max_seq_length, num_labels = num_labels)
    elif exp_type == "ablation":
        model = AblationLongBert.from_pretrained(args.bert_model, args.seq_segments, args.max_seq_length, num_labels = num_labels)
    else:
        exit("invalid experiment type of: {}".format(exp_type))

    return model


def load_dataset_old():
     # Prepare data loader
    logger.info("Loading training dataset")
    train_examples = processor.get_train_examples(args.data_dir)
    cached_train_features_file = os.path.join(args.data_dir, 'train_{0}_{1}_{2}_{3}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(),
                    str(args.max_seq_length),
                    str(task_name),
                    str(args.seq_segments)))
    try:
        with open(cached_train_features_file, "rb") as reader:
            gc.disable()
            train_features = pickle.load(reader)
            gc.enable()
    except:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, args.seq_segments, tokenizer, output_mode)
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("  Saving train features into cached file %s", cached_train_features_file)
            with open(cached_train_features_file, "wb") as writer:
                pickle.dump(train_features, writer)

    if args.task_name == "section" or args.task_name == "subclass":
        temp = None

    logger.info("Tensoring the training dataset")
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    elif output_mode == "regression"  or output_mode == "multi_classification":
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)
    
    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

def load_dataset(main_file, args, processor, tokenizer, output_mode, train = True):
     # Prepare data loader
    id_file     = main_file + "_ids.pt"
    mask_file   = main_file + "_masks.pt"
    seg_file    = main_file + "_segs.pt"
    label_file  = main_file + "_labels.pt"

    #case1 tensor files exist
    file_exist_count = sum([isfile(id_file), isfile(mask_file),isfile(seg_file),isfile(label_file)])
    if  0 < file_exist_count  < 4:
        exit("Only part of the data is saved as tensor files. Delete those files and try again.")
    elif file_exist_count == 4:
        all_input_ids = torch.load(id_file)
        all_input_mask = torch.load(mask_file)
        all_segment_ids = torch.load(seg_file)
        all_label_ids = torch.load(label_file)
        return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    #case2 old file exists
    if isfile(main_file):
        with open(main_file, "rb") as reader:
            features = pickle.load(reader)
    else:
        #load the mega object
        if train:
            features = convert_examples_to_features(
                processor.get_train_examples(args.data_dir), 
                processor.get_labels(), args.max_seq_length, args.seq_segments, tokenizer, output_mode)
        else:
            features = convert_examples_to_features(
                processor.get_dev_examples(args.data_dir), 
                processor.get_labels(), args.max_seq_length, args.seq_segments, tokenizer, output_mode)

    #parse it carefully into tensor files
    torch.save(torch.tensor([f.input_ids    for f in features], dtype=torch.long), id_file)
    torch.save(torch.tensor([f.input_mask   for f in features], dtype=torch.long), mask_file)
    torch.save(torch.tensor([f.segment_ids  for f in features], dtype=torch.long), seg_file)
    torch.save(torch.tensor([f.label_id     for f in features], dtype=torch.long), label_file)
    #call this function again!
    return load_dataset(main_file, args, processor, tokenizer, output_mode)



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
    parser.add_argument("--bert_model_path", default="", type=str, required=False,
                        help="Bert pretrained saved pytorch model path.")
    parser.add_argument("--experiment", default="attention", type=str, required=False,
                        help="4 types: attention, base, long, ablation. "
                        "base: original bert"
                        "long: uses an lstm to keep track of all bert hidden representations, but backprop over the first"
                        "attention: uses an lstm + attention mechanism to backprop over more than the first representation"
                        "ablation: concat all the hidden representations"
                        )
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--seq_segments",
                        default=8,
                        type=int,
                        help="The number of sequence steps")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_shuffle",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--super_debug",
                        action='store_true',
                        help="hack for debugging.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
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
    parser.add_argument('--overwrite_output_dir',
                        action='store_true',
                        help="Overwrite the content of the output directory")
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
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    args.device = device

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

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

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    
    cls_token = tokenizer.convert_tokens_to_ids(["[CLS]"])
    sep_token = tokenizer.convert_tokens_to_ids(["[SEP]"])

    '''if args.super_debug:
        cached_eval_features_file = os.path.join(args.data_dir, 'dev_{0}_{1}_{2}_{3}'.format(
                list(filter(None, args.bert_model.split('/'))).pop(),
                            str(args.max_seq_length),
                            str(task_name),
                            str(args.seq_segments)))

        logger.info("Loading test dataset")
        eval_data =  load_dataset(cached_eval_features_file, args, processor, tokenizer, output_mode, train = False)
        exit()'''
    #model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels = num_labels)
    #model = MyBertForMultiLabelSequenceClassification.from_pretrained(args.bert_model, num_labels = num_labels)
    model = get_model(args, num_labels)

    if args.bert_model_path != "":
        print("Loading model from: " + args.bert_model_path)
        if args.do_train:
            pretrained_dict = torch.load(args.bert_model_path)
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            '''if 'classifier.weight' in pretrained_dict and pretrained_dict['classifier.weight'].shape[0] != num_labels:
                del pretrained_dict['classifier.weight']
                del pretrained_dict['classifier.bias']
            if 'classifier2.weight' in pretrained_dict and pretrained_dict['classifier2.weight'].shape[0] != num_labels:
                del pretrained_dict['classifier2.weight']
                del pretrained_dict['classifier2.bias']'''
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            model.load_state_dict(model_dict)
        else:
            model.load_state_dict(torch.load(args.bert_model_path))
    
    sig = Sigmoid()
    if args.local_rank == 0:
        torch.distributed.barrier()

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
   
    loss_fct = CrossEntropyLoss()
    if args.do_train:
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()


        cached_train_features_file = os.path.join(args.data_dir, 'train_{0}_{1}_{2}_{3}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(),
                    str(args.max_seq_length),
                    str(task_name),
                    str(args.seq_segments)))

        # Prepare data loader
        logger.info("Loading training dataset")
        train_data =  load_dataset(cached_train_features_file, args, processor, tokenizer, output_mode)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        

        num_train_optimization_steps = (len(train_dataloader)) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
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
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

        logger.info("***** Running training *****")
        #logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        model.train()
        for i in trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, t_batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
                input_ids, input_mask, segment_ids, label_ids = t_batch
                if args.do_shuffle:
                    shuffled_index  = torch.randperm(input_ids.shape[0])

                    shuffled_ids    = input_ids[shuffled_index][:,:256]
                    shuffled_mask   = input_mask[shuffled_index][:,:256]
                    shuffled_seg    = segment_ids[shuffled_index][:,:256]

                    input_ids[:,:256] = shuffled_ids
                    input_mask[:,:256] = shuffled_mask
                    segment_ids[:,:256] = shuffled_seg


                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)


                logits = model(input_ids, segment_ids, input_mask)

                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1)) 
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()

                    global_step += 1
                    if args.local_rank in [-1, 0]:
                        acc = np.sum(np.argmax(logits.cpu().detach().numpy(), axis=1) == label_ids.cpu().numpy()) / label_ids.shape[0]
                        tb_writer.add_scalar('lr', optimizer.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', loss.item(), global_step)
                        tb_writer.add_scalar('acc', acc, global_step)

    ### Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    ### Example:
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        #model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

        # Good practice: save your training arguments together with the trained model
        output_args_file = os.path.join(args.output_dir, 'training_args.bin')
        torch.save(args, output_args_file)
        open(os.path.join(args.output_dir, 'experiment_{}.txt'.format(args.experiment)), 'a').close()
    else:
        model = get_model(args, num_labels)
        model.load_state_dict(torch.load(output_model_file))
        model.to(device)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)


    ### Evaluation
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        cached_eval_features_file = os.path.join(args.data_dir, 'dev_{0}_{1}_{2}_{3}'.format(
            list(filter(None, args.bert_model.split('/'))).pop(),
                        str(args.max_seq_length),
                        str(task_name),
                        str(args.seq_segments)))

        logger.info("Loading test dataset")
        eval_data =  load_dataset(cached_eval_features_file, args, processor, tokenizer, output_mode, train = False)
        #import pdb; pdb.set_trace()
        logger.info("***** Running evaluation *****")
        #logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        # Run prediction for full data
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)  # Note that this sampler samples randomly
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        out_label_ids = None

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)
               
            
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))


            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if output_mode == "multi_classification":
                logits =  sig(logits)

            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
                out_label_ids = label_ids.detach().cpu().numpy()
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)
        elif output_mode == "multi_classification":
            preds = preds > .5
        result = compute_metrics(task_name, preds, out_label_ids)

        loss = tr_loss/global_step if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss

        output_eval_file = os.path.join(args.output_dir, "eval_results_final.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

       

if __name__ == "__main__":
    main()
