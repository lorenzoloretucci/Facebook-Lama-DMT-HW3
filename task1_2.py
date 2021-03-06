# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from modules import build_model_by_name
from utils import print_sentence_predictions, load_vocab
import options
import codecs 

import torch
import numpy as np
import scipy

#########
# HOW TO RUN THIS SCRIPT

# FROM TERMINAL, INSIDE THE LAMA REPOSITORY

# python lama/task1_2.py  --lm "bert" --t path_mashed_sentences

# Ps: indisde the script you have to modify some paths...
#########

results = []  # LIST CONTAINING BERT MODEL PREDICTIONS

def __max_probs_values_indices(masked_indices, log_probs, topk=1000): # UNCHANGED

    # score only first mask
    masked_indices = masked_indices[:1]

    masked_index = masked_indices[0]
    log_probs = log_probs[masked_index]

    value_max_probs, index_max_probs = torch.topk(input=log_probs,k=topk,dim=0)
    index_max_probs = index_max_probs.numpy().astype(int)
    value_max_probs = value_max_probs.detach().numpy()

    return log_probs, index_max_probs, value_max_probs


def __print_top_k(value_max_probs, index_max_probs, vocab, mask_topk, index_list, max_printouts = 10): # UNCHANGED
    result = []
    msg = "\n| Top{} predictions\n".format(max_printouts)
    for i in range(mask_topk):
        filtered_idx = index_max_probs[i].item()

        if index_list is not None:
            # the softmax layer has been filtered using the vocab_subset
            # the original idx should be retrieved
            idx = index_list[filtered_idx]
        else:
            idx = filtered_idx

        log_prob = value_max_probs[i].item()
        word_form = vocab[idx]

        if i < max_printouts:
            msg += "{:<8d}{:<20s}{:<12.3f}\n".format(
                i,
                word_form,
                log_prob
            )
        element = {'i' : i, 'token_idx': idx, 'log_prob': log_prob, 'token_word_form': word_form}
        result.append(element)
    return result, msg

###
## WE MODIFIED THE FOLLOWING FUNCTION (GET_RANKING), IN ORDER TO GET THE TOP 10 PREDICTIONS! 
###

def get_ranking(log_probs, masked_indices, vocab, label_index = None, index_list = None, topk = 10, print_generation=True):
    experiment_result = {}

    log_probs, index_max_probs, value_max_probs = __max_probs_values_indices(masked_indices, log_probs, topk=topk)
    result_masked_topk, return_msg = __print_top_k(value_max_probs, index_max_probs, vocab, topk, index_list)
    experiment_result['topk'] = result_masked_topk

    outt = []
    for element in experiment_result['topk']:
        outt.append(element['token_word_form'])
    return outt




def main(args):

    if not args.text and not args.interactive:
        msg = "ERROR: either you start LAMA eval_generation with the " \
              "interactive option (--i) or you pass in input a piece of text (--t)"
        raise ValueError(msg)

    stopping_condition = True

    models = {}
    for lm in args.models_names:
        models[lm] = build_model_by_name(lm, args)
        print()

    vocab_subset = None
    if args.common_vocab_filename is not None:
        common_vocab = load_vocab(args.common_vocab_filename)
        print("common vocabulary size: {}".format(len(common_vocab)))
        vocab_subset = [x for x in common_vocab]

    texts = args.text
    f = codecs.open(texts, 'r', 'ISO-8859-1')
    for text in f:
        sentences = [text]
        out = ""
        for model_name, model in models.items():
            
            original_log_probs_list, [token_ids], [masked_indices] = model.get_batch_generation([sentences], try_cuda=False)

            index_list = None
            if vocab_subset is not None:
                # filter log_probs
                filter_logprob_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)
                filtered_log_probs_list = model.filter_logprobs(original_log_probs_list, filter_logprob_indices)
            else:
                filtered_log_probs_list = original_log_probs_list

            # rank over the subset of the vocab (if defined) for the SINGLE masked tokens
            if masked_indices and len(masked_indices) > 0:
                out = get_ranking(filtered_log_probs_list[0], masked_indices, model.vocab, index_list=index_list)
        results.append(out)

if __name__ == '__main__':
    parser = options.get_eval_generation_parser()
    args = options.parse_args(parser)
    main(args)


# FILE CONTAINING TRUE VALUE OF MASKS. 
path_true_mask = "/Users/yves/Desktop/Data_Science/first_year/second_semester/DMT/HW3/mask_values.txt" 

# FILE CONTAINING THE LABELS
path_label = '/Users/yves/Desktop/Data_Science/first_year/second_semester/DMT/HW3/label.txt'

mask_val = []
with open(path_true_mask, "r") as f: 
    for line in f:
        mask_val.append(line.strip("\n"))

label_val =[]
with open(path_label, "r") as f: 
    for line in f:
        label_val.append(line.strip("\n"))

support = 0
not_support = 0

# HEURISTIC OPTIMIZZATION  ==> SET TO LOWERCASE

final = []
for i in results:
    st = []
    for j in i:
        st.append(j.lower())
    final.append(st)
results = final

# ACCURACY EVALUATION
RES = []
for i in range(len(results)):
    if mask_val[i].lower() in results[i]: # Results contains the predicted. # check if the predicted word matches the true mention
        RES.append("SUPPORTS")
    else:
        RES.append("REFUTES")

for j in range(len(RES)):
    if RES[j] == label_val[j]:
        support += 1
    else:
        not_support += 1


print('TASK1_2 Results')
print()
print( "Support:    ", support)
print( "Reject:    ", not_support)

print("Accuracy:    ", str(round((support/(support + not_support)*100), 3)) + " %")
print()