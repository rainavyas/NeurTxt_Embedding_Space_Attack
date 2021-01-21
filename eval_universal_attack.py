''''
Reports statistics (MSE, PCC etc.) with a universal phrase added to each utterance
'''

import argparse
import json
from transformers import *
import sys
import os
import torch
from utility import *

# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('DATA', type=str, help='Specify input txt file with prepared useful data')
commandLineParser.add_argument('LOG', type=str, help='Specify file to log result')
commandLineParser.add_argument('GRADES', type=str, help='Specify grades file')
commandLineParser.add_argument('MODEL', type=str, help='Specify trained model path')
commandLineParser.add_argument('PHRASE', type=str, help='Phrase to append to each utterance')

args = commandLineParser.parse_args()
data_file = args.DATA
log_file = args.LOG
grades_file = args.GRADES
trained_model_path = args.MODEL
phrase = args.PHRASE

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/eval_universal_attack.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

MAX_UTTS_PER_SPEAKER_PART = 1
MAX_WORDS_IN_UTT = 200

# Load the data
with open(data_file, 'r') as f:
    utterances = json.loads(f.read())
print("Loaded Data")

# Convert json output from unicode to string
utterances = [[str(item[0]), str(item[1])] for item in utterances]

# Add the universal phrase to every utterance
utterances = [[item[0], item[1]+' '+phrase] for item in utterances]

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
model = torch.load(trained_model_path)
model.eval()

# Get model predictions
y_pred = model(X, M).squeeze()
y_pred[y_pred>6] = 6
y_pred[y_pred<0] = 0

# Get stats
avg = torch.mean(y_pred)
mse = calculate_mse(y.tolist(), y_pred.tolist())
pcc = calculate_pcc(y_pred, y)
less1 = calculate_less1(y_pred, y)
less05 = calculate_less05(y_pred, y)

# Save the stats
output = "Avg: " +str(avg) +"\nMSE: "+str(mse)+"\nPCC: "+str(pcc)+"\nLess than 1: "+str(less1)+"\nLess than 0.5: "+str(less05)
with open(log_file, 'w') as f:
    f.write("Attack Phrase: " + phrase+"\n")
    f.write(output)
print(output)
