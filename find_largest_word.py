'''
Looks through word logs and returns the one with the greatest cosine distance score
'''

import argparse
import sys
import os
import scandir

# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('DIR', type=str, help='Specify directory with all word log files')

args = commandLineParser.parse_args()
words_dir = args.DIR

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/find_largest_word.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

best = [None, 0]

# Get list of files in directory
files = [f.name for f in scandir.scandir(words_dir)]

for curr_file in files:
    print("Processing " + curr_file)
    curr_path = words_dir+"/"+curr_file
    with open(curr_path, 'r') as f:
        lines = f.readlines()
    last = lines[-1]
    print(last)
    items = last.split()
    word = str(items[0])
    cos_dist = float(items[1])
    if cos_dist > best[1]:
        best = [word, cos_dist]    

print(best)

