import pickle
import sys
import os
import argparse
import torch
from model_embedding_attack import Bespoke_Attack

def scale_params(model, e):
    old_params = {}

    for name, params in model.named_parameters():
        old_params[name] = params.clone()

    T = old_params['attack']
    T_abs = torch.abs(T)
    T_shaped = torch.reshape(T_abs, (T_abs.size(0), -1))
    largest_elements_ind = torch.max(T_shaped, dim=-1)
    largest_elements = torch.FloatTensor(largest_elements_ind[0])
    largest_elements = largest_elements.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    largest_elements = largest_elements.repeat(1, T.size(1), T.size(2), T.size(3))
    scaled_T = (T/largest_elements)*e
    old_params['attack'] = scaled_T

    for name, params in model.named_parameters():
        params.data.copy_(old_params[name])


# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('DATA', type=str, help='Specify input pkl file with prepared useful data')
commandLineParser.add_argument('MODEL', type=str, help='Specify trained model path')
commandLineParser.add_argument('OUT', type=str, help='Specify the output pt file to save model to')
commandLineParser.add_argument('--epsilon', default=1.0, type=float, help='Specify the epsilon constraint')

args = commandLineParser.parse_args()
data_file = args.DATA
trained_model_path = args.MODEL
out_file = args.OUT
e = args.epsilon

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/bespoke_adv_attack.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

# Load the data
pkl = pickle.load(open(data_file, "rb"))
X = pkl['X']
M = pkl['M']
speaker_ids = pkl['ids']

# Convert to tensors
X = torch.FloatTensor(X)
M = torch.FloatTensor(M)


# Define constants
lr = 2*1e-1
seed = 1
torch.manual_seed(seed)

# Define the attack tensor for each speaker in one tensor
attack_init = torch.zeros(X.size(0), X.size(1), X.size(2), X.size(3))

# Initialise Model
attack_model = Bespoke_Attack(trained_model_path, attack_init)
attack_model.train()

# Learn all bespoke attack vectors together
optimizer = torch.optim.SGD(attack_model.parameters(), lr=lr)


# forward pass
y_pred = attack_model(X, M)
y_pred[y_pred> 6.0]=6.0

# Compute loss
loss = -1*torch.sum(y_pred)

# Zero gradients, backward pass
optimizer.zero_grad()
loss.backward(retain_graph=True)
optimizer.step()

# Scale attack values to have an infinite norm equal to e
scale_params(attack_model, e)

no_attack_avg = (-1)*loss/X.size(0)
print("Average Grade with no attacks: ", no_attack_avg)

   
# Get the predicted grades with updated parameters
y_pred = attack_model(X, M)
y_pred[y_pred> 6.0]=6.0
attack_avg = torch.mean(y_pred)
print("Average Grade with linear attack: ", attack_avg)

# Attempt to predict a universal attack from averaging bespoke attacks
old_params = {}

# Save the model
torch.save(attack_model, out_file)

