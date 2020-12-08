import torch
from models import ThreeLayerNet_1LevelAttn_RNN

class Attack(torch.nn.Module):
    def __init__(self, embedding_size=768, num_words=200, trained_model_path, init_attack):

        super(Attack, self).__init__()
        self.trained_model_path = trained_model_path
        self.attack = torch.nn.Parameter(init_attack, requires_grad=True)

    def forward(self, X, M):
        '''
        X = [N x U x W x n]
        N = batch size
        U = number of utterances
        W = number of words
        n = number of features for word embedding
        '''
        attack_batched = torch.tile(self.attack, (X.size(0),1,1,1))
        X_attacked = X + attack_batched

        # Pass through trained model
        trained_model = torch.load(self.trained_model_path)
        trained_model.eval()
        y = trained_model(X_attacked, M)

        return y
