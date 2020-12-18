import torch
from model_embedding_attack import *
from models import ThreeLayerNet_1LevelAttn_RNN
import pickle
import sys
import os
import argparse
import matplotlib as plt
import seaborn as sns
from utility import calculate_pcc
import torch

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
    sns_plot.get_figure().clf()
    return y

def plot_attack_dot_product_dist(model_indv, model_uni, out_file_path):
    '''
    normalises attack vectors from both models first
    '''
    old_params_indv = {}
    old_params_uni = {}

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
             bins=60, color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
    sns_plot.set(xlabel="Cosine Similarity between universal and individual attacks")
    #sns_plot.set(xlim=(0, 6))
    sns_plot.figure.savefig(out_file_path)
    sns_plot.get_figure().clf()

    return d

def plot_scatter(x, y, xlabel, ylabel, out_file_path):
    # Calculate the pcc
    pcc = calculate_pcc(torch.FloatTensor(x), torch.FloatTensor(y))
    sns.set_theme(color_codes=True)
    scatter_plot =  sns.regplot(x=x, y=y)
    scatter_plot.set(xlabel=xlabel)
    scatter_plot.set(ylabel=ylabel)
    scatter_plot.set(title="PCC: "+str(pcc))
    scatter_plot.figure.savefig(out_file_path)
    scatter_plot.get_figure().clf()

def plot_change_in_y_vs_cosine(y_increase, cosine_d, out_file_path):
    sns.set_theme(color_codes=True)
    scatter_plot =  sns.regplot(x=cosine_d, y=y_increase)
    scatter_plot.set(xlabel="Cosine distance between individual and universal attacks")
    scatter_plot.set(ylabel="Increase in score from universal attack")
    scatter_plot.figure.savefig(out_file_path)
    scatter_plot.get_figure().clf()


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
model_indv_opt = torch.load(model_indv_opt_path)
model_indv_opt.eval()
model_uni_indv = torch.load(model_uni_indv_path)
model_uni_indv.eval()
model_uni_opt = torch.load(model_uni_opt_path)
model_uni_opt.eval()

# Plot the y value distributions
out_file = "y_no_attack.png"
y_no_attack = plot_y_dist(model, X, M, out_file)
out_file = "y_attack_ind_opt.png"
y_indv_opt = plot_y_dist(model_indv_opt, X, M, out_file)
out_file = "y_attack_uni_ind.png"
y_uni_indv = plot_y_dist(model_uni_indv, X, M, out_file)
out_file = "y_attack_uni_opt.png"
y_uni_opt = plot_y_dist(model_uni_opt, X, M, out_file)

# Plot the dot product distributions
out_file = "cosine_indv_opt_and_uni_indv.png"
cosine_indv_opt_and_uni_indv = plot_attack_dot_product_dist(model_indv_opt, model_uni_indv, out_file)
out_file = "cosine_indv_opt_and_uni_opt.png"
cosine_indv_opt_and_uni_opt = plot_attack_dot_product_dist(model_indv_opt, model_uni_opt, out_file)

# Plot scatter plots of y_pred pairings and calculate pcc
xlabel = "y with optimal individual attacks"
ylabel = "y with universal attack indv"
out_file = "scatter_y_indv_opt_and_y_uni_indv.png"
plot_scatter(y_indv_opt, y_uni_indv, xlabel, ylabel, out_file)

xlabel = "y with optimal individual attacks"
ylabel = "y with optimal universal attack"
out_file = "scatter_y_indv_opt_and_y_uni_opt.png"
plot_scatter(y_indv_opt, y_uni_opt, xlabel, ylabel, out_file)

xlabel = "y with optimal universal attack"
ylabel = "y with universal attack indv"
out_file = "scatter_y_uni_opt_and_y_uni_indv.png"
plot_scatter(y_uni_opt, y_uni_indv, xlabel, ylabel, out_file)

# Plot scatter plot of cosine distances from each universal attack
xlabel = "Cosine dist of opt indv attack with optimal universal attack"
ylabel = "Cosine dist of opt indv attack with universal attack indv"
out_file = "scatter_cosine_distances.png"
plot_scatter(cosine_indv_opt_and_uni_opt, cosine_indv_opt_and_uni_indv, xlabel, ylabel, out_file)

# Compute the cosine distance between universal attacks
old_params1 = {}
old_params2 = {}

for name, params in model_uni_indv.named_parameters():
    old_params1[name] = params.clone()
for name, params in model_uni_opt.named_parameters():
    old_params2[name] = params.clone()

delta1 = old_params1['attack']
delta2 = old_params2['attack']

delta1 = delta1.reshape(-1)
delta2= delta2.reshape(-1)

delta1_norm = delta1/torch.sqrt(torch.sum(delta1**2))
delta2_norm = delta2/torch.sqrt(torch.sum(delta2**2))
cosine_simalirity = torch.sum(delta2_norm*delta1_norm)
print("Cosine similarity between universal attack vectors: ", cosine_simalirity)

# Plot of increase in y against cosine distance
increase_y = torch.FloatTensor(y_attack_uni_opt)-torch.FloatTensor(y_no_attack)
increase_y.tolist()
cosine_d = cosine_indv_opt_and_uni_opt
out_file = "increase_y_vs_cosine_uni_opt.png"
plot_change_in_y_vs_cosine(y_increase, cosine_d, out_file)
