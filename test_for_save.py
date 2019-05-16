import torch
from torch.autograd import Variable
import  torch.nn.functional as F
import  matplotlib.pyplot as plt

n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), 0).type(torch.LongTensor)

x, y = Variable(x), Variable(y)

def save():


    # plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy())
    # plt.show()

    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)
            self.predict = torch.nn.Linear(n_hidden, n_output)

        def forward(self, x):
            x = F.relu(self.hidden(x))
            x = self.predict(x)
            return x

    net = Net(2, 10, 2)
    # print(net)

    # plt.ion()
    # plt.show()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
    loss_func = torch.nn.CrossEntropyLoss()

    for t in range(100):
        out = net(x)
        loss = loss_func(out, y)
        optimizer.zero_grad()  # zero the last group of gradient diviriate
        loss.backward()
        optimizer.step()

    out = net(x)
    prediction = torch.max(F.softmax(out), 1)[1]
    pred_y = prediction.data.numpy().squeeze()
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=10, lw=0, cmap='RdYlGn')


    # torch.save(net,'net.pkl') # entire net
    torch.save(net.state_dict(),'net_params.pkl') # parameters

def restore_net():
    net = torch.load('net.pkl')
    out = net(x)
    prediction = torch.max(F.softmax(out), 1)[1]
    pred_y = prediction.data.numpy().squeeze()
    plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')

def restore_params():
    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)
            self.predict = torch.nn.Linear(n_hidden, n_output)

        def forward(self, x):
            x = F.relu(self.hidden(x))
            x = self.predict(x)
            return x

    net = Net(2, 10, 2)
    net.load_state_dict(torch.load('net_params.pkl'))
    out = net(x)

    prediction = torch.max(F.softmax(out), 1)[1]
    pred_y = prediction.data.numpy().squeeze()
    plt.subplot(132)
    plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=10, lw=0, cmap='RdYlGn')
    plt.show()

save()
# restore_net()
restore_params()



# It seems that the torch.save is not avalible in this exaple, in the future, maybe I can solve it.