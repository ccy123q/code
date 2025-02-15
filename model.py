import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from statsmodels.tsa.seasonal import STL



class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class DTAN(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2):
        super(DTAN, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.rest_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        self.avg_linear1 = nn.Conv2d(in_channels=2, out_channels=2,
                                            kernel_size=(1, 1),bias=True)
        self.avg_linear2 = nn.Conv2d(in_channels=12, out_channels=13,
                                            kernel_size=(1, 1), bias=True)
        self.avg_linear3 = nn.Conv2d(in_channels=206, out_channels=207,
                                            kernel_size=(1, 1), bias=True)
        self.max_linear1 =  nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1, 1),bias=True)
        self.max_linear2 = nn.Conv2d(in_channels=12, out_channels=13,
                                            kernel_size=(1, 1), bias=True)
        self.max_linear3 = nn.Conv2d(in_channels=206, out_channels=207,
                                            kernel_size=(1, 1), bias=True)
        self.E_linear1 =  nn.Linear(in_features=1664, out_features=10,
                                      bias=True)
        self.attemd_linear1 = nn.Linear(in_features=13, out_features=13, bias=True)
        self.self_atten1 = nn.MultiheadAttention(embed_dim=32, num_heads=32)
        self.self_atten2 = nn.MultiheadAttention(embed_dim=32, num_heads=32)
        self.att_linear1 = nn.Conv2d(in_channels=207, out_channels=32, kernel_size=(1, 1), bias=True)
        self.att_linear2 = nn.Conv2d(in_channels=32, out_channels=207, kernel_size=(1, 1), bias=True)
        self.att_linear3 = nn.Conv2d(in_channels=13, out_channels=32, kernel_size=(1, 1), bias=True)
        self.att_linear4 = nn.Conv2d(in_channels=32, out_channels=13, kernel_size=(1, 1), bias=True)
        self.att_linear5 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1, 1), bias=True)
        self.att_linear6 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1, 1), bias=True)

        self.layernorm1 = nn.LayerNorm(normalized_shape=13)
        self.layernorm2 = nn.LayerNorm(normalized_shape=13)
        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))
                self.tcnlinear1 = nn.Conv2d(in_channels=residual_channels, out_channels=residual_channels,
                                            kernel_size=(1, 1), bias=True)
                self.tcnlinear2 = nn.Conv2d(in_channels=residual_channels, out_channels=residual_channels,
                                            kernel_size=(1, 1), bias=True)
                self.tcnlinear3 = nn.Conv2d(in_channels=residual_channels, out_channels=residual_channels,
                                            kernel_size=(1, 1), bias=True)
                self.tcnlinear4 = nn.Conv2d(in_channels=residual_channels, out_channels=residual_channels,
                                            kernel_size=(1, 1), bias=True)
                self.tcnlinear5 = nn.Conv2d(in_channels=residual_channels, out_channels=residual_channels,
                                            kernel_size=(1, 1), bias=True)
                self.tcnlinear6 = nn.Conv2d(in_channels=residual_channels, out_channels=residual_channels,
                                            kernel_size=(1, 1), bias=True)
                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.gcnlinear1 = nn.Conv2d(in_channels=residual_channels, out_channels=residual_channels,
                                            kernel_size=(1, 1), bias=True)
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):  # (64,2,207,13)
        kernel_size = (2, 2)
        stride = (1, 1)
        padding = (0, 0)  # 在高度和宽度方向都进行填充以保持高度和宽度不变
        avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        avg =avg_pool(input)#(64,2,206,12)
        max =max_pool(input)
        avg = avg.swapaxes(1, 3) #(64,12,206,2)
        avg=self.avg_linear2(avg)
        avg = avg.swapaxes(1, 2)#(64,12,206,2)
        avg=self.avg_linear3(avg)
        max = max.swapaxes(1, 3)
        max = self.max_linear2(max)
        max = max.swapaxes(1, 2)
        max = self.max_linear3(max)
        avg = avg.contiguous().view(207, 2, 64, 13)
        max = max.contiguous().view(207, 2, 64, 13)
        #
        #

        avg=(self.avg_linear1(avg))
        max =(self.max_linear1(max))

        attemd1 = input.permute(0,1,3,2)
        attemd1 = attemd1.contiguous().view(128, 13, 207)
        attemd1 = attemd1.permute(2, 1, 0)
        attemd1 = self.att_linear1(attemd1)
        attemd1 = attemd1.permute(2, 1, 0)#(128, 13,32)
        attemd1, _ =self.self_atten1(attemd1,attemd1,attemd1)
        attemd1 = attemd1.permute(2, 1, 0)
        attemd1 = self.att_linear2(attemd1)
        attemd1 = attemd1.contiguous().view(207, 2, 64, 13)
        attemd2 = input.contiguous().view(207, 128,13)
        attemd2 = attemd2.permute(2, 1, 0)
        attemd2 = self.att_linear3(attemd2)
        attemd2 = attemd2.permute(2, 1, 0)
        attemd2, _ = self.self_atten2(attemd2,attemd2,attemd2)
        attemd2 = attemd2.permute(2, 1, 0)
        attemd2 = self.att_linear4(attemd2)
        attemd2 = attemd2.contiguous().view(207, 2, 64, 13)
        attemd = self.att_linear5(attemd1)+self.att_linear6(attemd2)
        # attemd = self.att_linear6(attemd1)
        # em =  avg + max
        em = attemd+avg+max
        E=em.reshape(207,-1)
        E=self.E_linear1(E)
        ET = E.t()

        in_len = input.size(3)  # (64,2,207,13)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)  # (64,32,207,13)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports1 = None
        new_supports2 =None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp1 = F.softmax(F.relu(torch.mm(E, ET)), dim=1)
            adp2 = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports1 = self.supports + [ adp1]
            new_supports2 = self.supports+ [ adp2]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            gate = self.gate_convs[i](residual)
            r1 = torch.sigmoid(self.tcnlinear1(filter) + self.tcnlinear2(gate))
            z1 = torch.sigmoid(self.tcnlinear3(filter) + self.tcnlinear4(gate))
            ht2 = self.tcnlinear5(gate) + self.tcnlinear6(r1 * filter)
            ht1 = F.relu(ht2)
            x = z1 * ht1 + (1 - z1) * (filter)
            # x=filter
            # (64,32,207,12)
            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                space = x
                if self.addaptadj:

                    xgcn1 = self.gconv[i](space, new_supports1)  # (64,32,207,12)
                    xgcn2 = self.gconv[i](space, new_supports2)
                    x = F.relu(self.gcnlinear1(xgcn1+xgcn2))


                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)  # (64,32,207,1)
        x = F.relu(self.end_conv_1(x))  # (64,256,207,1)
        x = self.end_conv_2(x)  # (64,512,207,1)
        return x  # (64,12,207,1)





