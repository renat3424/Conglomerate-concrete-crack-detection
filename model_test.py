import torch
from torchsummary import summary
from utils import plot_result1, test_accuracy, ImageDataSet, DataLoader
from models import UNet, SegNet


if __name__ == '__main__':
    model=UNet(3)
    summary(model.cuda(), (3, 448,448))
    model.load_state_dict(torch.load("unet_model_weights90-0.0001.pth"))
    print(test_accuracy(model).item())
    batch_size=1
    path_str = "C:\\Users\\Ренат\\Downloads\\Conglomerate Concrete Crack Detection\\Conglomerate Concrete Crack Detection"
    test_dataset = ImageDataSet(path_str, train=True)
    test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    for i, (x, y) in enumerate(test):


        z = y[0].squeeze(0).to("cpu")
        z1 = x[0].permute(1, 2, 0).to("cpu")

        plot_result1(z, z1)
        if i==10:
            break



