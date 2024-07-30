import torch
import torch.nn as nn
import torch.nn.functional as F



class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_probability, up=False):
        super(UNetBlock, self).__init__()
        self.up=up
        if self.up:
            self.conv_transpose=nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2,2), stride=2)
        in_channels=2*out_channels if up else in_channels
        self.conv2d1=nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding="same")
        self.conv2d2=nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding="same")
        self.dropout=nn.Dropout2d(p=dropout_probability)
        self.pooling=nn.MaxPool2d(2, stride=2)
        self.relu=nn.ReLU()

    def forward(self, x, y=None):
        c=x
        if self.up:
            c=self.conv_transpose(x)
            c=torch.concatenate((c, y), dim=1)
        c=self.conv2d1(c)
        c=self.relu(c)
        c = self.dropout(c)
        c = self.conv2d2(c)
        return self.relu(c)

class UNet(nn.Module):
    def __init__(self, in_channels):
        super(UNet, self).__init__()
        self.pooling = nn.MaxPool2d(2, stride=2)
        self.block1=UNetBlock(in_channels, 16, 0.1)
        self.block2=UNetBlock(16, 32, 0.1)
        self.block3=UNetBlock(32, 64, 0.2)
        self.block4 = UNetBlock(64, 128, 0.2)
        self.block5=UNetBlock(128, 256, 0.3)
        self.block6=UNetBlock(256, 128, 0.2, True)
        self.block7 = UNetBlock(128, 64, 0.2, True)
        self.block8 = UNetBlock(64, 32, 0.1, True)
        self.block9 = UNetBlock(32, 16, 0.1, True)
        self.conv2d = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 1), padding="same")
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        c1=self.block1(x)
        p1=self.pooling(c1)

        c2 = self.block2(p1)
        p2 = self.pooling(c2)

        c3 = self.block3(p2)
        p3 = self.pooling(c3)

        c4 = self.block4(p3)
        p4 = self.pooling(c4)

        c5 = self.block5(p4)

        c6=self.block6(c5, c4)

        c7 = self.block7(c6, c3)

        c8=self.block8(c7, c2)

        c9=self.block9(c8, c1)

        return self.sigmoid(self.conv2d(c9))




class SegNet(nn.Module):

    def __init__(self):
        super(SegNet, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)

        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)

        self.deconv5_1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.deconv5_2 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.deconv5_3 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.deconv4_1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.deconv4_2 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.deconv4_3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.deconv3_1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.deconv3_2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.deconv3_3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2_1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2_2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv1_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv1_2 = nn.ConvTranspose2d(64, 2, kernel_size=3, stride=1, padding=1)

        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(512)

    def forward(self, x):
        size_1 = x.size()
        x = self.conv1_1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x, idxs1 = self.pool1(x)

        size_2 = x.size()
        x = self.conv2_1(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x, idxs2 = self.pool2(x)

        size_3 = x.size()
        x = self.conv3_1(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.conv3_3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x, idxs3 = self.pool3(x)

        size_4 = x.size()
        x = self.conv4_1(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.conv4_3(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x, idxs4 = self.pool4(x)

        size_5 = x.size()
        x = self.conv5_1(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.conv5_2(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.conv5_3(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x, idxs5 = self.pool5(x)

        x = self.unpool5(x, idxs5, output_size=size_5)
        x = self.deconv5_1(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.deconv5_2(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.deconv5_3(x)
        x = self.batch_norm4(x)
        x = F.relu(x)

        x = self.unpool4(x, idxs4, output_size=size_4)
        x = self.deconv4_1(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.deconv4_2(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.deconv4_3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)

        x = self.unpool3(x, idxs3, output_size=size_3)
        x = self.deconv3_1(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.deconv3_2(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.deconv3_3(x)
        x = self.batch_norm2(x)
        x = F.relu(x)

        x = self.unpool2(x, idxs2, output_size=size_2)
        x = self.deconv2_1(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.deconv2_2(x)
        x = self.batch_norm1(x)
        x = F.relu(x)

        x = self.unpool1(x, idxs1, output_size=size_1)
        x = self.deconv1_1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.deconv1_2(x)

        return x