import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter as Parameter


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class LRMC(nn.Module):
    def __init__(self,mip):
        super(LRMC, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv2 = nn.Conv2d(1, mip, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(mip)
        self.act2 = h_swish()

        self.conv_h = nn.Conv2d(mip, 1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, st, ss):
        map = st
        map2 = ss # （1，20，23，23）

        n, c, h, w = st.size()
        st_trans = st.squeeze(0).unsqueeze(1) # (20,1,23,23)
        ss_trans = ss.squeeze(0).unsqueeze(1) #
        x_h = self.pool_h(st_trans) # (20,1,23,1)
        x_w = self.pool_w(st_trans).permute(0, 1, 3, 2) # (20,1,1,23)
        x_h2 = self.pool_h(ss_trans) # (20,1,23,1)
        x_w2 = self.pool_w(ss_trans).permute(0, 1, 3, 2) # (20,1,23,1)

        y = torch.cat([x_h, x_h2], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        y2 = torch.cat([x_w, x_w2], dim=2)
        y2 = self.conv2(y2)
        y2 = self.bn2(y2)
        y2 = self.act2(y2)

        x_h, x_h2 = torch.split(y, [h, w], dim=2)
        x_w, x_w2 = torch.split(y2, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        x_w2 = x_w2.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid().squeeze(1).unsqueeze(0)
        a_w = self.conv_w(x_w).sigmoid().squeeze(1).unsqueeze(0)
        a_h2 = self.conv_h(x_h2).sigmoid().squeeze(1).unsqueeze(0)
        a_w2 = self.conv_w(x_w2).sigmoid().squeeze(1).unsqueeze(0)

        out = map * a_w * a_h + map2 * a_w2 * a_h2

        return out

class LRMC_conv(nn.Module):
    def __init__(self,mip):
        super(LRMC_conv, self).__init__()
        self.conv1 = nn.Conv2d(40, 20, kernel_size=1, stride=1, padding=0)


    def forward(self, st, ss):
        map = st
        map2 = ss # （1，20，23，23）

        map_c = torch.cat([map,map2], dim=1)
        out = self.conv1(map_c)
        return out

class LRMC_cross_att(nn.Module):
    def __init__(self):
        super(LRMC_cross_att, self).__init__()
        self.conv1=nn.Conv2d(20, 20, kernel_size=1, stride=1, padding=0)
        self.conv2=nn.Conv2d(20, 20, kernel_size=1, stride=1, padding=0)


    def forward(self, st, ss): # (1, 20, 23, 23)
        b, c, h, w = st.size()
        st = self.conv1(st)
        ss = self.conv2(ss)
        st_t = st.view(b, c, h*w)
        # st_t_T = st_t.permute(0,2,1)
        ss_t = ss.view(b, c, h * w)
        ss_t_T = ss_t.permute(0, 2, 1)
        M = torch.matmul(st_t, ss_t_T)
        M = F.softmax(M, dim=-1)
        out = torch.matmul(M, st_t).contiguous().view(b,c,h,w)
        return out

