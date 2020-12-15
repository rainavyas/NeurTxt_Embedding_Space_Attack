import torch
from model_embedding_attack import *
import pickle
import sys
import os
import argparse

# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('MODEL_INDV_OPT', type=str, help='Specify trained individual optimal attack model path')
commandLineParser.add_argument('MODEL_UNI', type=str, help='Specify trained individual optimal attack model path')
commandLineParser.add_argument('epsilon', type=float, help='Specify the epsilon constraint in the models used')



args = commandLineParser.parse_args()
model_indv_opt_path = args.MODEL_INDV_OPT
model_uni_path = args.MODEL_UNI
e = args.epsilon

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/get_uni_indv_model.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

# Load each model
model_indv_opt = torch.load(model_indv_opt)
model_uni = torch.load(model_uni_path)

# Copy the parameters
old_params = {}
for name, params in model_indv_opt.named_parameters():
    old_params[name] = params.clone()

attacks = old_params['attack']
avg_attack = torch.mean(attacks, dim=0)
old_params['attack'] = avg_attack
# Copy to a model of universal attack structure
for name, params in model_uni.named_parameters():
    params.data.copy_(old_params[name])

# Save the new universal attack model
outfile = "uni_indv_attack_epsilon"+str(e)+".pt"
torch.save(model_uni, outfile)
