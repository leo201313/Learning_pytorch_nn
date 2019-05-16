import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import  matplotlib.pyplot as plt


LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1,1,1000),dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(x.size()))

# plt.scatter(x.numpy(),y.numpy(),s=10)
# plt.show()

torch_dataset = Data.TensorDataset(x,y)
loader = Data.DataLoader(dataset=torch_dataset,batch_size=BATCH_SIZE,shuffle=True)

class Mynet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(1,20)
        self.predict = torch.nn.Linear(20,1)
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net_SGD = Mynet()
net_Momentum = Mynet()
net_RMSprop = Mynet()
net_Adam = Mynet()

net_ls = [net_SGD,net_Momentum,net_RMSprop,net_Adam]

opt_SGD = torch.optim.SGD(net_SGD.parameters(),lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))

opt_ls = [opt_SGD,opt_Momentum,opt_RMSprop,opt_Adam]

loss_func = torch.nn.MSELoss()
loss_history = [[],[],[],[]]

for epoch in range(EPOCH):
    print(epoch)
    for step,(x_batch,y_batch) in enumerate(loader):
        x_b = Variable(x_batch)
        y_b = Variable(y_batch)

        for net,opt,l_h in zip(net_ls,opt_ls,loss_history):
            output = net(x_b)
            loss = loss_func(output,y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_h.append(loss.data)

labels=['SGD','Momentum','RMSprop','Adam']
for i,l_h in enumerate(loss_history):
    plt.plot(l_h,label=labels[i])

plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0,0.2))
plt.show()


