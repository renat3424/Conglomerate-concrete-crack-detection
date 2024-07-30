import torch.optim as optim
from torchvision.models.segmentation import deeplabv3
from torchsummary import summary
from models import UNet, SegNet
from utils import ImageDataSet, DataLoader, plot_result, device, train_model
from _functional import IOU


if __name__=="__main__":
    path_str="C:\\Users\\Ренат\\Downloads\\Conglomerate Concrete Crack Detection\\Conglomerate Concrete Crack Detection"
    batch_size=4
    train_dataset=ImageDataSet(path_str)
    train=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)


    model=UNet(3).to(device)
    # model = deeplabv3.deeplabv3_resnet50(weights=None, pretrained=False, num_classes=1).to(device)
    print(summary(model.cuda(), (3, 448,448)))
    Loss = IOU()

    optimizer=optim.Adam(model.parameters(), lr=0.0001)

    step_lr_schedular=optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_model(model, train, Loss, optimizer, step_lr_schedular, num_epochs=21)

    #draw pictures

    test_dataset=ImageDataSet(path_str, train=False)
    test=DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    d=iter(test)
    x, y=next(d)
    x=x.to(device)
    y=y.to(device)
    y_pred=model(x[0].unsqueeze(0))["out"]

    z = y[0].detach().squeeze(0).to("cpu").numpy()
    z1 = x[0].detach().permute(1, 2, 0).to("cpu").numpy()
    z2 = y_pred.detach().squeeze(0).squeeze(0).to("cpu").numpy()
    plot_result(z2, z, z1)



    # print(y_pred)
    # print(y)
    # Loss=JaccardLoss(mode=BINARY_MODE, from_logits=False)
    # loss=Loss(y_pred, y).item()
    # print(loss, 1-loss)
    # y_pred=y_pred>0.5
    # loss = Loss(y_pred.to(torch.float32), y).item()
    # print(1-loss)
    # print(accuracy(y_pred, y))


    # Loss=nn.MSELoss()
    # optimizer=optim.Adam(model.parameters())
    # print("here")
    # print(x.shape)
    # y_pred=model(x)
    # print(y.shape, y_pred.shape)
    # for i, (x, y) in enumerate(train):
    #     x = x.to(device)
    #     y=y.to(device)
    #     y_pred = model(x)
    #     loss=Loss(y_pred, y)
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     print(loss.item())

    # d=iter(train)
    # x, y=next(d)
    # x=x.to(device)
    # y=y.to(device)



        