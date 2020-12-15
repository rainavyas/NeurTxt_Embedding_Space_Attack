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

def plot_attack_dot_product_dist(model_indv, model_uni, out_file_path):
    '''
    normalises attack vectors from both models first
    '''
    old_params_indv = {}
    old_params_uni

    for name, params in model_indv.named_parameters():
        old_params_indv[name] = params.clone()
    for name, params in model_uni.named_parameters():
        old_params_uni[name] = params.clone()

    delta_indv = old_params_indv['attack']
    delta_uni = old_params_uni['attack']

    delta_indv = delta_indv.reshape(delta_indv.size(0), -1)
    delta_uni = delta_uni.reshape(-1)

    delta_indv_norm = delta_indv/(torch.sqrt(torch.sum(delta_indv**2, dim=1, keepdim=True)).repeat((1,delta_indv.size(-1))))
    delta_uni_norm = delta_uni/torch.sqrt(torch.sum(delta_uni**2))

    cosine_simalirity = torch.sum(delta_indv_norm*(delta_uni_norm.repeat((delta_indv_norm.size(0), 1))), dim=1)
    d = cosine_simalirity.tolist()

    sns_plot = sns.distplot(d, hist=True, kde=True,
             bins=int(6/0.1), color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
    sns_plot.set(xlabel="Cosine Similarity between universal and individual attacks")
    #sns_plot.set(xlim=(0, 6))
    sns_plot.figure.savefig(out_file_path)



# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('DATA', type=str, help='Specify input pkl file with prepared useful data')
commandLineParser.add_argument('MODEL', type=str, help='Specify trained model path')
commandLineParser.add_argument('MODEL_INDV_OPT', type=str, help='Specify trained individual optimal attack model path')
commandLineParser.add_argument('MODEL_UNI_INDV', type=str, help='Specify trained individual inspired universal attack model path')
commandLineParser.add_argument('MODEL_UNI_OPT', type=str, help='Specify trained optimal universal attack model path')
commandLineParser.add_argument('--epsilon', default=None, type=float, help='Specify the epsilon constraint in the models used')

args = commandLineParser.parse_args()
data_path = args.DATA
model_path = args.MODEL
model_indv_opt_path = args.MODEL_INDV_OPT
model_uni_indv_path = args.MODEL_UNI_INDV
model_uni_opt_path = args.MODEL_UNI_OPT
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
out_file = "y_no_attack.png"
plot_y_dist(model, X, M, out_file)
out_file = "y_attack_ind_opt.png"
plot_y_dist(model_indv_opt, X, M, out_file)
out_file = "y_attack_uni_ind.png"
plot_y_dist(model_uni_indv, X, M, out_file)
out_file = "y_attack_uni_opt.png"
plot_y_dist(model_uni_opt, X, M, out_file)

# Plot the dot product distributions
out_file = "cosine_indv_opt_and_uni_indv.png"
plot_attack_dot_product_dist(model_indv_opt, model_uni_indv, out_file)
out_file = "cosine_indv_opt_and_uni_indv.png"
plot_attack_dot_product_dist(model_indv_opt, model_uni_opt, out_file)
