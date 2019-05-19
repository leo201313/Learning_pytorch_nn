import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = False

train_data = dsets.MNIST(
    root = './mnist',
    train = True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

test_data = dsets.MNIST(root='./mnist/',train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data,dim=1)).type(torch.FloatTensor)[:2000]/255
test_y = test_data.test_labels[:2000]

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.out = nn.Linear(64,10)

    def forward(self, x):
        r_out,(h_n,h_c) = self.rnn(x,None)
        out = self.out(r_out[:,-1,:])
        return out

rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x = Variable(x.view(-1,28,28))
        b_y = Variable(y)
        output = rnn(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x.view(-1,28,28))
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = sum(pred_y == test_y.numpy()) / test_y.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

    test_output = rnn(test_x[:10].view(-1,28,28))
    pred_y = torch.max(test_output,1)[1].data.numpy()
    print(pred_y,'prediction number')
    print(test_y[:10].numpy(),'real number')