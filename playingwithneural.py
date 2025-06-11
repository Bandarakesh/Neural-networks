import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.datasets import make_moons, make_circles
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
print(y.shape)
print(y.values.shape)
# print(X.head(5))
# print(y.head(5))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# vectorizer=CountVectorizer()
# Xvectorized=vectorizer.fit_transform(X).toarray()

def docstring():
    """we are actually developing a neural network  wine quality  datasets and see how neurons , layers ,
 activation functions impact"""
    pass
class MyNeural(nn.Module):
    def __init__(self,input_dim):
        super(MyNeural,self).__init__()
        self.h1=nn.Linear(input_dim,64)
        self.h2=nn.Linear(64,32)
        self.h3=nn.Linear(32,16)
        self.h4=nn.Linear(16,8)
        self.h5=nn.Linear(8,1)
        self.relu=nn.ReLU()
    
    def forward(self,x):
        x=self.relu(self.h1(x))
        x=self.relu(self.h2(x))
        x=self.relu(self.h3(x))
        x=self.relu(self.h4(x))
        x=self.h5(x)
        return x
class WineDataset(Dataset):
    def __init__(self,X,y):
        self.X=torch.tensor(X.values,dtype=torch.float32)
        self.y=torch.tensor(y.values,dtype=torch.float32).squeeze()
    def __getitem__(self, idx):
        return self.X[idx],self.y[idx]
    def __len__(self):
        return len(self.X)

traindataset=WineDataset(X_train,y_train)
testdataset=WineDataset(X_test,y_test)

train_loader=DataLoader(traindataset,shuffle=True,batch_size=64)
testloader=DataLoader(testdataset,shuffle=True,batch_size=64)
def feedforward():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on: {device}")
    input_dim=X.shape[1]

    model=MyNeural(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(10):
        model.train()
        for batch_X ,batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output=model(batch_X)
            loss = criterion(output.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        model.eval()
        val_loss=0
        while torch.no_grad():
            for X_batch,y_batch in testloader:
                X_batch=X_batch.to(device)
                y_batch=y_batch.to(device)
                pred=model(X_batch)
                val_loss+=criterion(pred.squeeze(),y_batch)
        val_loss /= len(testloader)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

feedforward()