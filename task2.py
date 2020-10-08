# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
from lama.modules import build_model_by_name
import lama.options as options
import argparse
import pandas as pd
import numpy as np
from transformers import BertTokenizer
import torch

from sklearn import svm
from sklearn.svm import SVC 
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import jsonlines


path_train_claims = "/Users/yves/Desktop/Data_Science/first_year/second_semester/DMT/HW3/Original_train_sentence.txt"
path_train_masked = "/Users/yves/Desktop/Data_Science/first_year/second_semester/DMT/HW3/mask_values_train.txt"
path_train_labels = "/Users/yves/Desktop/Data_Science/first_year/second_semester/DMT/HW3/Label_train.txt"


path_test_claims = "/Users/yves/Desktop/Data_Science/first_year/second_semester/DMT/HW3/test_sentence.txt"
path_test_masked = "/Users/yves/Desktop/Data_Science/first_year/second_semester/DMT/HW3/test_sentence_masked.txt"



# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# BERT model parameters
PARAMETERS= {
        "lm": "bert",
        "bert_model_name": "bert-base-cased",
        "bert_model_dir":
        "/Users/yves/Desktop/Data_Science/first_year/second_semester/DMT/HW3/LAMA/bert/cased_L-12_H-768_A-12",
        "bert_vocab_name": "vocab.txt",
        "batch_size": 32
        }
args = argparse.Namespace(**PARAMETERS)
models = {}
models['bert'] = build_model_by_name('bert', args)


# FUNCTION TO CREATE THE CONTEXTUAL EMBEDDINGS.
# It takas as input a list containing both the mask and the unmasked claim
# ex. [["Lorelai Gilmore's father is named [MASK]."],["Lorelai Gilmore's father is named Robert."]] 

def embed(sentences):
    for model_name, model in models.items():
        contextual_embeddings, sentence_lengths, tokenized_text_list = model.get_contextual_embeddings(sentences)

        rows = []

        pos = 0
        for i in range(len(tokenized_text_list)):
            if "[MASK]" in tokenized_text_list[i]:
                pos = tokenized_text_list[i].index("[MASK]")
            rows.extend(contextual_embeddings[11][i][pos])  # concatenate vectors
        
        return rows

        
def Sentences(path_mask, path_normal):  # Fucntion to build the sentences that are going to be used for the contextual embendding. 
                                        # The first list will contain the masked sentence and the sencond list, the unmasked claim
    masked = []
    with open(path_mask, "r") as f: 
        for line in f:
            masked.append(line.strip("\n"))

    normal = []
    with open(path_normal, "r") as f: 
        for line in f:
            normal.append(line.strip("\n"))

    CONCAT = []
    for i in range(len(masked)):
        CONCAT.append([[masked[i]], [normal[i]]])

    return CONCAT


def Get_labels(path_lab):  # Get the labels from file
    lab = []
    with open(path_lab, "r") as f: 
        for line in f:
            lab.append(line.strip("\n"))
    return lab

# Build dataFrame
def DataFrame(masked, normal):   # Build the matrix from the vectors coming form the contextual embedding
    a = Sentences(masked, normal)
    Rows = []
    for line in a:
        Rows.append(embed(line))

    out = pd.DataFrame(Rows)
    return out


X_train = DataFrame(path_train_masked, path_train_claims)  # Training set
y_train = Get_labels(path_train_labels)    # Training Labels

X_test = DataFrame(path_test_masked, path_test_claims)     # Testing set


param_grid =  {'C': [1000],  'gamma': [0.0001],'kernel': ['rbf'], 'degree': [10]} # BEST Parameters for the GridSearch
 
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,scoring = 'accuracy', cv= 7) # GridSearch
 
#Train the model using the training sets
grid.fit(X_train, y_train)
 
#Predict the response for test dataset
y_pred = grid.predict(X_test)


# Save the output to file
test_path = "/Users/yves/Desktop/Data_Science/first_year/second_semester/DMT/HW3/singletoken_test_fever_homework_NLP.jsonl"

data = []
with jsonlines.open(test_path) as f:
    for line in f:
        data.append(line['id'])


with jsonlines.open("Prediction_task2.jsonl", "w") as outfile:
    for i in range(len(data)):
        line = {'id': data[i], 'label': y_pred[i]}
        outfile.write(line)
