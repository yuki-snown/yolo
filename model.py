import torch
import torch.nn as nn

class DarknetConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
    super(DarknetConv,self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=bias)
    self.bn = nn.BatchNorm2d(out_channels)
    self.dropout = nn.Dropout2d(0.25)
    self.relu = nn.LeakyReLU(0.1)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.dropout(x)
    x = self.relu(x)
    return x

class ResBlock(nn.Module):
  def __init__(self, channels):
    super(ResBlock,self).__init__()
    self.conv1 = DarknetConv(channels, channels//2, 1)
    self.conv2 = DarknetConv(channels//2, channels, 3, padding=1)

  def forward(self,x):
    y = self.conv1(x)
    y = self.conv2(y)
    return torch.add(x,y)

class YOLO(nn.Module):
  def __init__(self):
    super(YOLO,self).__init__()
    self.conv1_1 = DarknetConv(3,32,3,padding=1)
    self.conv1_2 = DarknetConv(32,64,3,stride=2,padding=1)
    self.res_block2 = ResBlock(64)
    self.conv2 = DarknetConv(64,128,3,stride=2,padding=1)
    self.res_block3 = nn.ModuleList([ResBlock(128) for _ in range(2)])  
    self.conv3 = DarknetConv(128,256,3,stride=2,padding=1)
    self.res_block4 = nn.ModuleList([ResBlock(256) for _ in range(8)])
    self.conv4 = DarknetConv(256,512,3,stride=2,padding=1)    
    self.res_block5 = nn.ModuleList([ResBlock(512) for _ in range(8)])
    self.conv5 = DarknetConv(512,1024,3,stride=2,padding=1)
    self.res_block6 = nn.ModuleList([ResBlock(1024) for _ in range(4)])
    
  def forward(self,x):
    x = self.conv1_1(x) # (32, h, w)
    x = self.conv1_2(x) # (64, h/2, w/2)
    x = self.res_block2(x) # (64, h/2, w/2)

    x = self.conv2(x) # (128, h/4, w/4)
    for block in self.res_block3:
      x = block(x)
    _ = x # (128, h/4, w/4)

    # find small obj
    x = self.conv3(x) # (256, h/8, w/8)
    for block in self.res_block4:
      x = block(x)
    y1 = x # (256, h/8, w/8)

    # find midium obj
    x = self.conv4(x) # (512, h/16, w/16)
    for block in self.res_block5:
      x = block(x)
    y2 = x # (512, h/16, w/16)

    # find big obj
    x = self.conv5(x) # (1024, h/32, w/32)
    for block in self.res_block6:
      x = block(x)    
    y3 = x # (1024, h/32, w/32)

    return y1, y2, y3

class Detector(nn.Module):
  def __init__(self,cls_num=80):
    super(Detector,self).__init__()
    self.yolo = YOLO()
    self.cls_num = cls_num
    self.conv1_1 = DarknetConv(256, 128, 1)    
    self.conv1_2 = DarknetConv(384,256,3,padding=1)
    self.conv1_3 = nn.Conv2d(256, 3*(1+4+cls_num), 1)
    self.conv1 = nn.ModuleList([DarknetConv(384,128,1), 
                                DarknetConv(128,256,3,padding=1),
                                ResBlock(256),
                                DarknetConv(256,128,1)])

    self.conv2_1 = DarknetConv(512, 256, 1)
    self.conv2 = nn.ModuleList([DarknetConv(768,256,1), 
                                DarknetConv(256,512,3,padding=1),
                                ResBlock(512),
                                DarknetConv(512,256,1)])
    self.conv2_2 = DarknetConv(768,512,3,padding=1)
    self.conv2_3 = nn.Conv2d(512, 3*(1+4+cls_num), 1)

    self.conv3 = nn.ModuleList([ResBlock(1024),
                                ResBlock(1024),
                                DarknetConv(1024,512,1)])
    self.conv3_1 = DarknetConv(1024,1024,3,padding=1)
    self.conv3_2 = nn.Conv2d(1024, 3*(1+4+cls_num), 1)

  def forward(self, x):
    x = self.yolo(x)
    y3 = x[2]
    x3 = self.conv3_1(y3) # (1024, h/32, w/32)
    x3 = self.conv3_2(x3) # target -> x3 (_, h/32, w/32)
    for conv in self.conv3:
      y3 = conv(y3) # (512, h/32, w/32)

    y2 = self.conv2_1(y3) # (256, h/32, w/32)
    y2= nn.Upsample((x[1].size()[2], x[1].size()[3]))(y2) # (256, h/16, w/16)
    y2 = torch.cat([x[1], y2], dim=1) # (768, h/16, w/16)
    x2 = self.conv2_2(y2) # (512, h/16, w/16)
    x2 = self.conv2_3(x2) # target -> x2 (_, h/16, w/16)
    for conv in self.conv2:
      y2 = conv(y2) # (256, h/32, w/32)

    y1 = self.conv1_1(y2) # (128, h/16, w/16)
    y1= nn.Upsample((x[0].size()[2], x[0].size()[3]))(y1) # (128, h/8, w/8)
    y1 = torch.cat([x[0], y1], dim=1) # (384, h/8, w/8)
    x1 = self.conv1_2(y1) # (256, h/8, w/8)
    x1 = self.conv1_3(x1) # target -> x1 (_, h/8, w/8)

    # small, midium, large
    return [x1, x2, x3]