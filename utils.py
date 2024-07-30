import torch
from torch.utils.data import Dataset, DataLoader
import os.path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
from _functional import iou_accuracy


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def test_accuracy(model, smooth=1):
    batch_size=4
    path_str = "C:\\Users\\Ренат\\Downloads\\Conglomerate Concrete Crack Detection\\Conglomerate Concrete Crack Detection"
    test_dataset = ImageDataSet(path_str, train=False)
    test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    sum_intersections = 0
    sum_unities = 0
    with torch.no_grad():

        for i, (x,y) in enumerate(test):
            x=x.to("cuda")
            y=y.to("cuda")
            y_pred=model(x).detach()
            inter, unity=iou_accuracy(y, y_pred)
            sum_unities+=unity
            sum_intersections+=inter

    return  (sum_intersections+smooth)/(sum_unities+smooth)

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def plot_result1( y, x):
    # z = y[3].squeeze(0).to("cpu")
    # z1 = x[3].permute(1, 2, 0).to("cpu")
    # z2 = y_pred.squeeze(0).squeeze(0).to("cpu")

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(x)
    axarr[0].set(xlabel="original image")
    axarr[1].imshow(y, cmap="gray")
    axarr[1].set(xlabel="image")
def plot_result(y_pred, y, x):
    # z = y[3].squeeze(0).to("cpu")
    # z1 = x[3].permute(1, 2, 0).to("cpu")
    # z2 = y_pred.squeeze(0).squeeze(0).to("cpu")

    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(x)
    axarr[0].set(xlabel="original image")
    axarr[1].imshow(y, cmap="gray")
    axarr[1].set(xlabel="image")
    axarr[2].imshow(y_pred, cmap="gray")
    axarr[2].set(xlabel="predicted mask")
    plt.show()

def accuracy(y_pred, y):
    y_pred=y_pred>0.5
    y_pred=y_pred.to(torch.float32)
    return (y_pred==y).float().sum().item()/y.size().numel()

class ImageDataSet(Dataset):
    def __init__(self, path, train=True):
        self.data=[]
        self.target=[]
        path_str="Train" if train else "Test"
        self.data_path=os.path.join(path, path_str, "images")
        self.target_path=os.path.join(path, path_str, "masks")

        for file in os.listdir(self.data_path):
            file_path=os.path.join(self.data_path, file)
            self.data.append(file_path)

        for file in os.listdir(self.target_path):
            file_path = os.path.join(self.target_path, file)
            self.target.append(file_path)




    def __getitem__(self, index):
        temp = Image.open(os.path.join(self.data_path, self.data[index]))
        x=np.asarray(temp, dtype=np.float32)
        temp.close()

        temp = Image.open(os.path.join(self.target_path, self.target[index]))
        y = np.asarray(temp)
        if len(y.shape)>2:
            y=rgb2gray(y)
        y=y/255>0.95
        temp.close()
        return torch.Tensor(x).permute(2, 0 ,1)/255, torch.Tensor(y).unsqueeze(0)


    def __len__(self):
        return len(self.data)

def train_model(model, train_dataset, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    losses=[]
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for i, (x, y) in enumerate(train_dataset):
            x = x.to(device)
            y = y.to(device)
            y_pred=model(x)["out"]
            loss=criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"step {i + 1}, loss={loss.item()}, accuracy={1-loss.item()}")

        #scheduler.step()
        acc=test_accuracy(model).item()
        losses.append(acc)
        print("optimizer learning rate: ", optimizer.param_groups[0]['lr'], " test iou=", acc)
        if epoch%10==0:
            torch.save(model.state_dict(), 'some.pth')
    time_elapsed = time.time() - since

    np.save("test_some.npy", np.array(losses))
    print(f"elapsed time={time_elapsed}")