import pickle
import sys
import os
import argparse
import torch
from model_embedding_attack import Universal_Attack

def clip_params(model, e):
    old_params = {}

    for name, params in model.named_parameters():
        old_params[name] = params.clone()

    T = old_params['attack']
    T[T>e]=e
    T[T<(-1)*e]=(-1)*e
    old_params['attack'] = T

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
with open('CMDs/universal_adv_attack.cmd', 'a') as f:
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
epochs = 75
seed = 1
torch.manual_seed(seed)

# Define the attack tensor for each speaker in one tensor
attack_init = torch.zeros(X.size(1), X.size(2), X.size(3))

# Initialise Model
attack_model = Universal_Attack(trained_model_path, attack_init)
attack_model.train()

# Learn universal attack vector
optimizer = torch.optim.SGD(attack_model.parameters(), lr=lr, momentum = 0.9, nesterov=True)

original_avg = 0
for epoch in range(epochs):
    print("On epoch ", epoch)

    # forward pass
    y_pred = attack_model(X, M)
    y_pred[y_pred> 6.0]=6.0

    # Compute loss
    loss = -1*torch.sum(y_pred)

    # Zero gradients, backward pass
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    # Keep attack values below threshold
    clip_params(attack_model, e)

    avg = (-1)*loss/X.size(0)
    print("Average Grade with universal attack: ", avg)


    if epoch == 0:
        original_avg = avg

print("------------------------------------------------------------------------------------")
print("No Attack Average Grade: ", original_avg)
print("Universal Attack Average Grade ", avg)



# Save the model
torch.save(attack_model, out_file)
