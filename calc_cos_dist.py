import torch

'''
torch.manual_seed(10)
Cov = torch.randn(768, 768)

for i in range(768):
    for j in range(i):
        Cov[i,j] = Cov[j,i]

e, v = torch.symeig(Cov, eigenvectors=True)
e_abs = torch.abs(e)
inds = torch.argsort(e_abs)
e = e[inds]
v = v[inds]

#print(e)
#print(v[-1])

det = torch.det(Cov)
print(det)
'''
x1 = torch.load("temp1.pt")
x2 = torch.load("temp2.pt")

print(x1)

cos = torch.nn.CosineSimilarity(dim=0)
sim = cos(x1, x2)

print(sim)

