'''
Find the covariance matrix in the embedding space
Use each eigenvector direction as an attack
Generate plots of impact on mse, avg grade, pcc
Attack scaled using sign method to epsilon
'''

import pickle
import sys
import os
import argparse
import torch
from models import ThreeLayerNet_1LevelAttn
import matplotlib.pyplot as plt
from utility import *
import json
from transformers import *

def apply_attack(X, attack_direction, epsilon):
    attack_signs = torch.sign(attack_direction)
    attack = (attack_signs * epsilon) # can multiply by -1 to reverse direction of attack
    attack_batched = attack.repeat((X.size(0),1,1,1))
    X_attacked = X + attack_batched
    return X_attacked

commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('DATA', type=str, help='Specify input txt file with prepared useful data')
commandLineParser.add_argument('MODEL', type=str, help='Specify trained model path')
commandLineParser.add_argument('GRADES', type=str, help='Specify grades file')
commandLineParser.add_argument('--epsilon', default=1.0, type=float, help='Specify the epsilon constraint')

args = commandLineParser.parse_args()
data_file = args.DATA
trained_model_path = args.MODEL
grades_file = args.GRADES
epsilon = args.epsilon

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/generate_universal_variance_attacks.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

MAX_UTTS_PER_SPEAKER_PART = 1
MAX_WORDS_IN_UTT = 200

# Load the data
with open(data_file, 'r') as f:
    utterances = json.loads(f.read())
print("Loaded Data")

# Convert json output from unicode to string
utterances = [[str(item[0]), str(item[1])] for item in utterances]

# Load tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_basic_tokenize=False, do_lower_case=True)
bert_model = BertModel.from_pretrained('bert-base-cased')
bert_model.eval()
print("Loaded BERT model")

# Convert sentences to a list of BERT embeddings (embeddings per word)
# Store as dict of speaker id to utterances list (each utterance a list of embeddings)
utt_embs = {}

for item in utterances:
    fileName = item[0]
    speakerid = fileName[:12]
    sentence = item[1]

    tokenized_text = tokenizer.tokenize(sentence)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    if len(indexed_tokens) < 1:
        word_vecs = [[0]*768]
    else:
        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            outputs = bert_model(tokens_tensor)
            encoded_layers = outputs.last_hidden_state
            #encoded_layers, _ = bert_model(tokens_tensor)
        bert_embs = encoded_layers.squeeze(0)
        word_vecs = bert_embs.tolist()
    if speakerid not in utt_embs:
        utt_embs[speakerid] =  [word_vecs]
    else:
        utt_embs[speakerid].append(word_vecs)

# get speakerid to section grade dict
grade_dict = {}
section_num = 3

lines = [line.rstrip('\n') for line in open(grades_file)]
for line in lines[1:]:
        speaker_id = line[:12]
        grade_overall = line[-3:]
        grade1 = line[-23:-20]
        grade2 = line[-19:-16]
        grade3 = line[-15:-12]
        grade4 = line[-11:-8]
        grade5 = line[-7:-4]
        grades = [grade1, grade2, grade3, grade4, grade5, grade_overall]

        grade = float(grades[section_num-1])
        grade_dict[speaker_id] = grade


# Create list of grades and speaker utterance in same speaker order
grades = []
vals = []

for id in utt_embs:
    try:
        grades.append(grade_dict[id])
        vals.append(utt_embs[id])
    except:
        print(id)

# Convert to appropriate 4D tensor
# Initialise list to hold all input data in tensor format
X = []
y = []

# Initialise 2D matrix format to store all utterance lengths per speaker
utt_lengths_matrix = []

for utts, grade in zip(vals, grades):
        new_utts = []

        # Reject speakers with not exactly correct number of utterances in part
        if len(utts) != MAX_UTTS_PER_SPEAKER_PART:
                continue


        # Create list to store utterance lengths
        utt_lengths = []

        for curr_utt in utts:
                num_words = len(curr_utt)

                if num_words <= MAX_WORDS_IN_UTT:
                        # append padding of zero vectors
                        words_to_add = MAX_WORDS_IN_UTT - num_words
                        zero_vec_word = [0]*768
                        zero_vec_words = [zero_vec_word]*words_to_add
                        new_utt = curr_utt + zero_vec_words
                        utt_lengths.append(num_words)
                else:
                        # Shorten utterance from end
                        new_utt = curr_utt[:MAX_WORDS_IN_UTT]
                        utt_lengths.append(MAX_WORDS_IN_UTT)

                # Convert all values to float
                new_utt = [[float(i) for i in word] for word in new_utt]

                new_utts.append(new_utt)

        X.append(new_utts)
        y.append(grade)
        utt_lengths_matrix.append(utt_lengths)

# Convert to tensors
X = torch.FloatTensor(X)
y = torch.FloatTensor(y)
L = torch.FloatTensor(utt_lengths_matrix)

# Make the mask from utterance lengths matrix L
M = [[([1]*utt_len + [-100000]*(X.size(2)- utt_len)) for utt_len in speaker] for speaker in utt_lengths_matrix]
M = torch.FloatTensor(M)

# Load the trained model
trained_model = torch.load(trained_model_path)
trained_model.eval()

# Calculate covariance matrix

# Compute mean embedding per speaker
E = torch.sum(X, dim=2).squeeze()
L_repeated = L.repeat(1,768)
E = E/L_repeated
E_mean = torch.mean(E, dim=0)
E_mean_matrix = torch.outer(E_mean, E_mean)
E_corr_matrix = torch.matmul(torch.transpose(E, 0, 1), E)/E.size(0)
Cov = E_corr_matrix - E_mean_matrix

# Find eigenvectors and eigenvalues
e, v = torch.symeig(Cov, eigenvectors=True)
v = torch.transpose(v, 0, 1)
e_abs = torch.abs(e)
inds = torch.argsort(e_abs)
e = e[inds]
v = v[inds]

# Use each eigenvector as an attack direction in turn
eigenvalues = []
ranks = []
mses = []
pccs = []
avgs = []
cosines = [] # Do this part later
ratios = [] # Do this later too

for i in range(0, e.size(0), 10):

    eigenvalue = e[i]
    eigenvector = v[i]
    X_attacked = apply_attack(X, eigenvector, epsilon)
    y_pred = trained_model(X_attacked, M).squeeze()
    y_pred[y_pred>6] = 6
    y_pred[y_pred<0] = 0
    avg = torch.mean(y_pred)
    mse = calculate_mse(y.tolist(), y_pred.tolist())
    pcc = calculate_pcc(y_pred, y)

    ranks.append(i)
    eigenvalues.append(eigenvalue)
    mses.append(mse)
    pccs.append(pcc)
    avgs.append(avg)

# Plot the graphs

# MSE vs eigenvalue
plt.plot(eigenvalues, mses)
plt.ylabel("MSE")
plt.xlabel("Eigenvalue")
plt.xscale('log')
plt.savefig("mse_eigenvalue.png")
plt.clf()

# PCC vs eigenvalue
plt.plot(eigenvalues, pccs)
plt.ylabel("MSE")
plt.xlabel("Eigenvalue")
plt.xscale('log')
plt.savefig("pcc_eigenvalue.png")
plt.clf()

# avg vs eigenvalue
plt.plot(eigenvalues, avgs)
plt.ylabel("MSE")
plt.xlabel("Eigenvalue")
plt.xscale('log')
plt.savefig("avg_eigenvalue.png")
plt.clf()
