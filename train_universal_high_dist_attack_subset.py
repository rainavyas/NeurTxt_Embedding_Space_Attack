'''
Iterates through words in subset vocabulary to find the word that maximises
cosine distance between the change in mean embeddings and the largest
eigenvector direction of covariance matrix,
where variance is calculated across the different speakers
'''

import argparse
import json
from transformers import *
from datetime import date
import sys
import os
import torch

# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('DATA', type=str, help='Specify input txt file with prepared useful data')
commandLineParser.add_argument('VOCAB', type=str, help='Specify txt file with subset word list')
commandLineParser.add_argument('PHRASE', type=str, help='Specify existing greedy phrase to append to every utterance')

args = commandLineParser.parse_args()
data_file = args.DATA
vocab_file = args.VOCAB
phrase = args.PHRASE

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/train_universal_high_dist_attack_subset.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

MAX_UTTS_PER_SPEAKER_PART = 1
MAX_WORDS_IN_UTT = 200

# Load the data
with open(data_file, 'r') as f:
    utterances = json.loads(f.read())
#print("Loaded Data")

# Convert json output from unicode to string
utterances = [[str(item[0]), str(item[1])] for item in utterances]

# Get the list of words to iterate through
with open(vocab_file, 'r') as f:
    test_words = f.readlines()
test_words = [str(word.rstrip('\n')).lower() for word in test_words]
print(test_words[:10])

# Add empty word to test words
test_words = [''] + test_words

# Load tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_basic_tokenize=False, do_lower_case=True)
bert_model = BertModel.from_pretrained('bert-base-cased')
bert_model.eval()
print("Loaded BERT model")

# Define threshold to beat
best = ['none', 0]

for word_num, new_word in enumerate(test_words):

    # Add new word and existing phrase to every utterance
    if word_num==0:
        new_utterances = [[item[0], item[1]+' '+new_word] for item in utterances]
    else:
        new_utterances = [[item[0], item[1]+' '+phrase+' '+new_word] for item in utterances]

    # Convert sentences to a list of BERT embeddings (embeddings per word)
    # Store as dict of speaker id to utterances list (each utterance a list of embeddings)
    utt_embs = {}
    for item in new_utterances:
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

    # Initialise list to hold all input data in tensor format
    X = []

    # Initialise 2D matrix format to store all utterance lengths per speaker
    utt_lengths_matrix = []

    for id in utt_embs:
        utts = utt_embs[id]
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
        utt_lengths_matrix.append(utt_lengths)

    XT = torch.FloatTensor(X)
    L = torch.FloatTensor(utt_lengths_matrix)

    # Make the mask from utterance lengths matrix L
    M = [[([1]*utt_len + [-100000]*(XT.size(2)- utt_len)) for utt_len in speaker] for speaker in utt_lengths_matrix]

    # Compute mean embedding per speaker
    E = torch.sum(XT, dim=2).squeeze()
    L_repeated = L.repeat(1,768)
    E = E/L_repeated

    # For no word added, determine attack direction
    if word_num == 0:
        E_original = E
    else:
        if new_word=='':
            continue
        # Determine average cosine distance of shifts to attack direction for all other words
        shift = E - E_original
        dist = torch.mean(shift**2, dim=1)
        avg_dist = torch.mean(dist)
        avg_dist = avg_dist.item()
        print(new_word, avg_dist)
        # Check if better than best
        if avg_dist > best[1]:
            best = [new_word, avg_dist]
            out = '\n'+best[0]+ " " + str(best[1])
            print("--------------------------------")
            print(out)
            print("--------------------------------")
