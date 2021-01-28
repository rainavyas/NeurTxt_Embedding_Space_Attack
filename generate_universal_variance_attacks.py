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

def apply_attack(X, attack_direction, epsilon):
    attack_signs = torch.sign(attack_direction)
    attack = (attack_signs * epsilon) # can multiply by -1 to reverse direction of attack
    attack_batched = self.attack.repeat((X.size(0),1,1,1))
    X_attacked = X + attack_batched
    return X_attacked

commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('DATA', type=str, help='Specify input pkl file with prepared useful data')
commandLineParser.add_argument('MODEL', type=str, help='Specify trained model path')
commandLineParser.add_argument('GRADES', type=str, help='Specify grades file')
commandLineParser.add_argument('--epsilon', default=1.0, type=float, help='Specify the epsilon constraint')

args = commandLineParser.parse_args()
data_file = args.DATA
trained_model_path = args.MODEL
grades_file = args.GRADES
e = args.epsilon

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/generate_universal_variance_attacks.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

# Load the data
pkl = pickle.load(open(data_file, "rb"))
X = pkl['X']
M = pkl['M']
speaker_ids = pkl['ids']

# Get the grades
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
for id in speaker_ids:
    grades.append(grade_dict[id])


X = torch.FloatTensor(X)
M = torch.FloatTensor(M)
y = torch.FloatTensor(grades)

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
