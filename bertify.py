import pickle
import sys
import os
import argparse
from transformers import *

# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('DATA', type=str, help='Specify input txt file with useful data')
commandLineParser.add_argument('OUT', type=str, help='Specify the output pkl file to store to')
commandLineParser.add_argument('--PART', default=3, type=int, help='Specify the section/part of the test')

args = commandLineParser.parse_args()
data_file = args.DATA
out_file = args.OUT
part = args.PART

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/bertify.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')


# Define the number of utterances per speaker per part
utt_part_vals = [6, 8, 1, 1, 5]
MAX_UTTS_PER_SPEAKER_PART = utt_part_vals[part-1]
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
            encoded_layers, _ = bert_model(tokens_tensor)
        bert_embs = encoded_layers.squeeze(0)
        word_vecs = bert_embs.tolist()
    if speakerid not in utt_embs:
        utt_embs[speakerid] =  [word_vecs]
    else:
        utt_embs[speakerid].append(word_vecs)

# Convert to appropriate 4D tensor
# Initialise list to hold all input data in tensor format
X = []

# Maintain list of speaker ids
speaker_ids = []

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
    speaker_ids.append(id)

# Make the mask from utterance lengths matrix L
M = [[([1]*utt_len + [-100000]*(X.size(2)- utt_len)) for utt_len in speaker] for speaker in utt_lengths_matrix]

# Store all the Bertified data and the associated speaker ids in pickle file
pkl_obj = {'X': X, 'M': M, 'ids': speaker_ids}
pickle.dump(pkl_obj, open(out_file, "wb"))
