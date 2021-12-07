# %%
import numpy as np
import pandas as pd
import pickle
from ipywidgets import interact
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# %%
df0 = pd.read_parquet('pass0.parquet')
df1 = pd.read_parquet('pass1.parquet')

def get_timestamps(row):
    # algin obd.spd_veh and acc
    # row = df0.iloc[k]
    mask = row['GM.T'] == 'acc.xyz'
    accT = row['GM.TS_or_Distance'][mask]
    obdT = row['GM.TS_or_Distance'][~mask]
    return accT, obdT

# %%
# print(list(df0))
df0.head()

# %%
def plot_loghist(x, bins):
    logbins = np.logspace(np.log10(min(x)),np.log10(max(x)),bins)
    plt.hist(x, bins=logbins)
    plt.xscale('log')
    plt.show()

# most recordings last ~2000 steps
zacc = df0['GM.acc.xyz.z']
plot_loghist(zacc.map(len), bins=20)

# %%
# distribution of IRI mean
iri = df0['IRI_mean']
plt.hist(iri, bins=20);

# %%
n = len(zacc)
nbs = list(np.random.randint(0,n,9))
nbs.sort(key=lambda k: iri.iloc[k])
nbs = np.array(nbs).reshape(3,3)

fig, ax = plt.subplots(3,3, figsize=(15,15))
for i in range(3):
    for j in range(3):
        k = nbs[i,j]
        ax1 = ax[i,j]
        ax1.plot(zacc.iloc[k])
        ax1.set_ylim(.8,1.2)
        ax1.set_title("k=%d IRI=%.5f" % (k, iri[k]))
        #ax2 = ax1.twinx().twiny()
        #ax2.plot(df0['IRI_sequence'].iloc[k], c='orange')

# %%
# snouglou
for _, row in df0.iterrows():
    riri = row['IRI_mean']
    c = 'red' if riri > 4 else 'lightgreen' if riri < 2 else 'blue' 
    plt.plot(row['GM.lat_int'], row['GM.lon_int'], c=c)

# %%
def mean_delta(seq):
    return np.abs(seq[1:] - seq[:-1]).mean()

def quantile_diff(seq, q=.15):
    return np.quantile(seq, 1-q) - np.quantile(seq, q)

a,b = zacc.map(mean_delta), zacc.map(quantile_diff)

fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].scatter(iri, a, s=.5)
ax[1].scatter(iri, b, s=.5);

# %%
# np.corrcoef(iri, a)
np.corrcoef(iri, b)
# IRI is correlated with the displacement

# %%
speed = df0['GM.obd.spd_veh.value']
min(speed.map(min)), max(speed.map(max))

# %%
dts = df0['GM.TS_or_Distance'].map(np.diff)
# dts = dts[dts.map(len) > 2048].map(lambda x: x[:2048]).reset_index(drop=True)
dts = np.concatenate(dts).astype('float') / 1e9
len(dts[dts > 1])

# %%
for _ in range(4):
    i = np.random.randint(0,n)
    s0 = df0.loc[i]
    accT, obdT = get_timestamps(s0)
    accZ = s0['GM.acc.xyz.z']
    obd = s0['GM.obd.spd_veh.value']
    # obd = gaussian_filter1d(obd, 3)
    
    # plt.plot(accZ)
    fig, ax1 = plt.subplots()
    ax1.set_title('k=%d IRI=%.3f' % (i,iri[i]))
    ax2 = ax1.twinx()
    ax1.plot(accT, accZ, color='#1f77b4')
    ax1.set_ylim(.8, 1.2)
    ax2.plot(obdT, obd, color='#ff7f0e')
    ax2.set_ylim(0, 65);

# we notice some clipping on the orange curve (?)

# %%
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TruncatedDS(Dataset):
    """
    In this dataset, all time series are truncated to the 0.1 quantile
    or size, if provided
    """
    def __init__(self, fname, size=None, align=32):
        df = pd.read_parquet(fname)
        zacc = df['GM.acc.xyz.z'] # Series[arr[:]float]
        iri = df['IRI_mean'] # Series[float]
        
        if not size:
            size = zacc.map(len).quantile(.1)
            size = (int(size) // align) * align
        self.sz = size
        mask = zacc.map(len) >= size
        zacc = zacc[mask]
        iri = iri[mask]
        
        def truncate(arr):
            return arr[:size]
        zacc = np.stack(zacc.map(truncate)).astype('float32')
        # print(zacc.shape, size)
        self.zacc = zacc
        self.iri = iri.values.astype('float32')
        
        print('TruncatedDS initialized with size = %d, len = %d' % (size,len(zacc)))
        
    
    def __getitem__(self, idx):
        return self.zacc[idx][None,:], self.iri[idx]
    
    def __len__(self):
        return len(self.zacc)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# https://keras.io/examples/timeseries/timeseries_anomaly_detection/
class AE1(nn.Module):
    def __init__(self, din=1, depth=2, w=7, h1=16, h2=1):
        super().__init__()
        assert depth >= 2
        pad = {'padding':'same', 'padding_mode':'reflect'}
        
        enc = [
            nn.Conv1d(din, h1, w, **pad),
            nn.MaxPool1d(2,2),
            nn.ReLU(),
            nn.Dropout(.2),
        ]
        dec = [
            nn.Conv1d(h1, h1, w, **pad),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(h1, din, w, **pad),
        ]
        for _ in range(2, depth):
            enc += [
                nn.Conv1d(h1, h1, w, **pad),
                nn.MaxPool1d(2,2),
                nn.ReLU(),
                nn.Dropout(.2),
            ]
            dec = [            
                nn.Conv1d(h1, h1, w, **pad),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Dropout(.2),
            ] + dec
        enc += [
            nn.Conv1d(h1, h2, w, **pad),
            nn.MaxPool1d(2,2),
            nn.ReLU(),
        ]
        dec = [
            nn.Conv1d(h2, h1, w, **pad),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Dropout(.2),
        ] + dec
        
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)
        
    def forward(self, x):
        encoded = self.encoder(x)
        # print(encoded.shape)
        decoded = self.decoder(encoded)
        return decoded

# %%
def get_trivial_mse(dataset, scale=8):
    def get_one(x):
        x,_ = x
        x1 = F.avg_pool1d(x,scale,scale)
        x1 = F.interpolate(x1, scale_factor=scale)
        return F.mse_loss(x1,x)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    return float(np.array(list(map(get_one, dataloader))).mean())

# %%
def train(model, dataset, epochs, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    hist = []
    for epoch in tqdm(range(epochs)):
        running_loss = 0
        for x,_ in dataloader:
            pred = model(x)
            loss = loss_fn(pred, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss) * x.size(0)

        hist.append(running_loss / len(dataset))
    return hist

# %%
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
# model = AE1()
# model(next(iter(dataloader)))

# %%
L = 2048
dataset = TruncatedDS('pass0.parquet', L)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# %%
models = []
def train_model(epochs=40, batch_size=4, **kwargs):
    model = AE1(**kwargs)
    hist = train(model, dataset, epochs, batch_size)
    models.append((model, hist, kwargs))

# %%
# train_model(depth=4, h2=4)
train_model(epochs=10, depth=2, h2=4)

# %%
scale = 4
print('compression x%d, trivial MSE: %f' % (scale, get_trivial_mse(dataset, scale)))
scale = 32
print('compression x%d, trivial MSE: %f' % (scale, get_trivial_mse(dataset, scale)))

# %%
def show_report(model):
    model, hist, _ = models[model]
    fig, ax = plt.subplots(1,2, figsize=(15,5))

    ax[0].plot(hist)
    ax[0].set_yscale('log')

    with torch.no_grad():
        x,iri = next(iter(dataloader))
        ax[1].plot(x[0,0].numpy())
        ax[1].plot(model(x)[0,0].numpy())
        ax[1].set_ylim(.8, 1.2)
        
interact(show_report, model=[((i,models[i][2]), i) for i in range(len(models))]);

# %%
## visualization
model = models[0][0]

for _ in range(3):
    i = np.random.randint(0,n)
    x = torch.tensor(zacc[i].astype('float32')).view(1,1,-1)
    with torch.no_grad():
        x = model.encoder(x).numpy()
    print("k=%d IRI=%.5f" % (i, iri[i]))
    plt.scatter(*x[0,2:], s=5);

# %%
## prediction

batch_size = 4
epochs = 4

rnn = nn.LSTM(input_size=4, hidden_size=32, num_layers=1, proj_size=1, batch_first=True)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)

hist = []
for epoch in (range(epochs)):
    running_loss = 0
    for x,iri in tqdm(dataloader):
        with torch.no_grad():
            z = model.encoder(x)
        y,_ = rnn(torch.transpose(z,1,2))
        loss = loss_fn(y[:,-1,0], iri)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)

        running_loss += float(loss) * x.size(0)

    hist.append(running_loss / len(dataset))


# %%
plt.plot(hist)

# %%
## prediction

batch_size = 4
epochs = 4

rnn = nn.LSTM(input_size=1, hidden_size=32, num_layers=1, proj_size=1, batch_first=True)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)

hist = []
for epoch in (range(epochs)):
    running_loss = 0
    for x,iri in tqdm(dataloader):
        y,_ = rnn(torch.transpose(x,1,2))
        loss = loss_fn(y[:,-1,0], iri)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)

        running_loss += float(loss) * x.size(0)

    hist.append(running_loss / len(dataset))


# %%
plt.plot(hist)

# %%
linreg = nn.Linear(1,1)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(linreg.parameters(), lr=1e-3)

hist = []
for epoch in tqdm(range(30)):
    running_loss = 0
    for x,iri in (dataloader):
        z = np.mean(np.abs(np.diff(np.array(x))), axis=-1)[:,None]
        z = torch.tensor(z)
        y = linreg(z)
        loss = loss_fn(y[:,-1,0], iri)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)

        running_loss += float(loss) * x.size(0)

    hist.append(running_loss / len(dataset))


# %%
plt.plot(hist);

# %%
list(linreg.parameters())

# %%



