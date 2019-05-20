import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

EPOCH = 10
BATCH_SIZE = 64
N_TEST_IMG = 5
LR = 0.005
DOWNLOAD_MNIST = False

train_data = dsets.MNIST(
    root = './mnist',
    train = True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(28*28,128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,12),
            nn.Tanh(),
            nn.Linear(12,3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3,12),
            nn.Tanh(),
            nn.Linear(12,64),
            nn.Tanh(),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded



autoencoder = AutoEncoder()
optimizer = torch.optim.Adam(autoencoder.parameters(),lr=LR)
loss_func = nn.MSELoss()

data_view = train_data.test_data[:N_TEST_IMG].view(-1,28*28).type(torch.FloatTensor)/255
figure,ax = plt.subplots(2,N_TEST_IMG,figsize=(5,2))
plt.ion()

for i in range(N_TEST_IMG):
    ax[0][i].imshow(np.reshape(data_view.data.numpy()[i], (28, 28)), cmap='gray')
    ax[0][i].set_xticks(())
    ax[0][i].set_yticks(())


for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x = Variable(x.view(-1,28*28))
        b_y = Variable(x.view(-1,28*28))
        b_label = Variable(y)

        encoded,decoded = autoencoder(b_x)

        loss = loss_func(decoded,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print('Eponch: ',epoch,'| train loss: %.4f' % loss.data.item())
            _,data_pre = autoencoder(data_view)
            for i in range(N_TEST_IMG):
                ax[1][i].imshow(np.reshape(data_pre.data.numpy()[i], (28, 28)), cmap='gray')
                ax[1][i].set_xticks(())
                ax[1][i].set_yticks(())
            plt.draw()
            plt.pause(0.05)

plt.ioff()
plt.show()

# View the plot in 3D
view_data = train_data.train_data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.
encoded_data, _ = autoencoder(view_data)
fig = plt.figure(2); ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = train_data.train_labels[:200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
plt.show()
