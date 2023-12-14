import torch
from pathlib import Path
import torch.nn.functional as F
import sys
data = sys.argv[1]
h = F.normalize(torch.arange(7).float(), 2, -1)

trainset = []
p = Path(f'/code/chenxi02/AAAI/data/{data}/dates')
for i in range(6):
    mp = p/str(i+1)/'matrix.pt'
    m = torch.load(mp)
    trainset.append(m)

tset = torch.cat(trainset,dim=0).cuda()
h = h - torch.mean(h)
label = h[:6].repeat(m.shape[0]).cuda()

class STJGC(torch.nn.Module):
    def __init__(self):
        super(STJGC,self).__init__()
        self.enc = torch.nn.Sequential(torch.nn.Linear(m.shape[1], 990),torch.nn.ReLU(),torch.nn.Linear(990,1000))
        self.dec = torch.nn.Sequential(torch.nn.Linear(1000,500),torch.nn.ReLU(),torch.nn.Linear(500,1))

    def forward(self):
        emb = self.enc(tset)
        output = self.dec(emb)
        return emb,output.squeeze(1)

model = STJGC().cuda()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 200
criterion = torch.nn.L1Loss()
# criterion = torch.nn.L1Loss()

label2 = torch.zeros(6,1000)
label2 = label2.cuda()
for i in range(epochs):
    optimizer.zero_grad()
    e,o = model()
    e = e.reshape(6,-1,1000)
    var = torch.var(e,dim=1)
    loss2 = criterion(var,label2)
    loss1 = criterion(o,label)
    # print(loss1)
    # print(loss2)
    loss = loss1+loss2        
    loss.backward()
    optimizer.step()

e,o = model()
ee = e.reshape(6,-1,1000)
ee = torch.mean(ee,1)
step = (torch.sum(ee[1:],0)-5*ee[0])/15
eeap = (ee[5]+step)
ee = torch.cat([ee,eeap.unsqueeze(0)],axis=0)
torch.save(ee, f'/code/chenxi02/AAAI/data/{data}/task3/time_embedding.pt')
print(f'/code/chenxi02/AAAI/data/{data}/task3/time_embedding.pt')