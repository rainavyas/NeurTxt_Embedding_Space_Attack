import torch
from model_embedding_attack import *
from models import ThreeLayerNet_1LevelAttn_RNN
import pickle
import sys
import os
import argparse
import matplotlib as plt
import seaborn as sns

def plot_y_dist(model, X, M, out_file_path):
    y_pred = model(X, M)
    y_pred = y_pred.squeeze()
    y_pred[y_pred>6.0]=6.0
    y_pred[y_pred<0.0]=0.0
    y = y_pred.tolist()
    sns_plot = sns.distplot(y, hist=True, kde=True,
             bins=int(6/0.1), color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
    sns_plot.set(xlabel="y")
    sns_plot.set(xlim=(0, 6))
    sns_plot.figure.savefig(out_file_path)



# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('DATA', type=str, help='Specify input pkl file with prepared useful data')
commandLineParser.add_argument('MODEL', type=str, help='Specify trained model path')
commandLineParser.add_argument('MODEL_INDV_OPT', type=str, help='Specify trained individual optimal attack model path')
commandLineParser.add_argument('MODEL_UNI_INDV', type=str, help='Specify trained individual inspired universal attack model path')
commandLineParser.add_argument('MODEL_UNI_OPT', type=str, help='Specify trained optimal universal attack model path')
commandLineParser.add_argument('OUT_PREFIX', type=str, help='Specify the output png files location and prefix')
commandLineParser.add_argument('--epsilon', default=None, type=float, help='Specify the epsilon constraint in the models used')

args = commandLineParser.parse_args()
data_path = args.DATA
model_path = args.MODEL
model_indv_opt_path = args.MODEL_INDV_OPT
model_uni_indv_path = args.MODEL_UNI_INDV
model_uni_opt_path = args.MODEL_UNI_OPT
out_file_prefix = args.OUT_PREFIX
e = args.epsilon

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/plot_distributions.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')


# Load the data
pkl = pickle.load(open(data_path, "rb"))
X = pkl['X']
M = pkl['M']
speaker_ids = pkl['ids']

# Convert to tensors
X = torch.FloatTensor(X)
M = torch.FloatTensor(M)

# Load the original and attack models
model = torch.load(model_path)
model.eval()
model_ind_opt = torch.load(model_indv_opt_path)
model_ind_opt.eval()
model_uni_indv = torch.load(model_uni_indv_path)
model_uni_indv.eval()
model_uni_opt = torch.load(model_uni_opt_path)
model_uni_opt.eval()

# Plot the y value distributions
out_file = "no_attack.png"
plot_y_dist(model, X, M, out_file)
