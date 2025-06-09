import math, torch, torch.nn as nn

def autopad(k, p=None):
    return k // 2 if p is None else p  # „same“‑Padding

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = nn.SiLU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1, self.cv2 = Conv(c1,c_,1), Conv(c_,c2,3)
        self.add           = shortcut and c1 == c2
    def forward(self,x):
        y = self.cv2(self.cv1(x))
        return x+y if self.add else y

class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1, self.cv2, self.cv3 = Conv(c1,c_,1), Conv(c1,c_,1), Conv(2*c_,c2,1)
        self.m = nn.Sequential(*[Bottleneck(c_,c_,shortcut,1.0) for _ in range(n)])
    def forward(self,x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)),1))

class SPPF(nn.Module):
    def __init__(self,c1,c2,k=5):
        super().__init__()
        c_ = c1//2
        self.cv1, self.cv2 = Conv(c1,c_,1), Conv(c_*4,c2,1)
        self.m = nn.MaxPool2d(k,1,k//2)
    def forward(self,x):
        x = self.cv1(x)
        y1 = self.m(x); y2 = self.m(y1); y3 = self.m(y2)
        return self.cv2(torch.cat((x,y1,y2,y3),1))

class YOLOv5Small(nn.Module):
    def __init__(self, num_classes: int = 1,
                 anchors=( (0.315,0.44, 0.565,0.69, 0.875,1.125),   # P3  stride 4
                (0.69,0.905, 1.125,1.405, 1.78,2.28),    # P4  stride 8
                (1.53,2.03, 2.845,3.78, 6.0,7.9) )): # P5  stride 16
        super().__init__()
        self.num_classes, self.num_outputs = num_classes, num_classes+5
        self.anchors = torch.tensor(anchors).float().view(3,-1,2)  # (3,3,2)

        # Backbone
        self.stem   = Conv(3,32,3,2)
        self.stage1 = Conv(32,64,3,2);  self.c3_1 = C3(64,64,1)
        self.stage2 = Conv(64,128,3,2); self.c3_2 = C3(128,128,3)
        self.stage3 = Conv(128,256,3,2);self.c3_3 = C3(256,256,3); self.spp = SPPF(256,256)

        # Neck / Head
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv4 = Conv(256+128,128,1); self.c3_4 = C3(128,128,3, shortcut=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv5 = Conv(128+64, 64,1);  self.c3_5 = C3(64, 64,3, shortcut=False)

        self.detect1 = nn.Conv2d(64,  self.num_outputs*3,1)
        self.down1   = Conv(64,128,3,2)
        self.c3_6    = C3(128+128,128,3, shortcut=False)
        self.detect2 = nn.Conv2d(128, self.num_outputs*3,1)
        self.down2   = Conv(256,256,3,2)
        self.c3_7    = C3(256+256,256,3, shortcut=False)
        self.detect3 = nn.Conv2d(256, self.num_outputs*3,1)

        self._init_bias()

    def _init_bias(self, prior=0.01):
        for m in (self.detect1,self.detect2,self.detect3):
            b = m.bias.view(3,-1)
            b.data[:,4] += math.log(2/(640/32)**2)  # obj
            b.data[:,5:] += math.log(prior/(1-prior))
            m.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self,x):
        x1 = self.c3_1(self.stage1(self.stem(x)))      # 1/4
        x2 = self.c3_2(self.stage2(x1))                # 1/8
        x3 = self.spp(self.c3_3(self.stage3(x2)))      # 1/16

        y1 = self.c3_4(self.cv4(torch.cat((self.up1(x3), x2),1)))
        y2 = self.c3_5(self.cv5(torch.cat((self.up2(y1), x1),1)))

        p3 = self.detect1(y2)
        p4 = self.detect2(self.c3_6(torch.cat((self.down1(y2), y1),1)))
        p5 = self.detect3(self.c3_7(torch.cat((self.down2(torch.cat((self.down1(y2), y1),1)), x3),1)))
        return [p3,p4,p5]
