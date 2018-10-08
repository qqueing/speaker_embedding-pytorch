
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLU20(nn.Hardtanh):
    def __init__(self, inplace=False):
        super(ReLU20, self).__init__(0, 20, inplace)
    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str


class gaoModel(nn.Module):
    def __init__(self,embedding_size,num_classes,input_dim = 64,hidden_unit=[512,512,512,512,512],hidden_use_p=[(40, 5,1), (1, 3,2), (1, 3,4),(1,1,1), (1,1,1)],bottle=[512]):
        super(gaoModel, self).__init__()

        self.embedding_size = embedding_size

        self.non_linear = ReLU20()

        self.layer = []
        for i,(unit,use_p) in enumerate(zip(hidden_unit,hidden_use_p)):
            concat_layer = conv_frame(input_dim, unit, use_p, non_linear=self.non_linear)
            input_dim = unit

            self.layer.append(concat_layer)
            setattr(self, 'cocat_layer{}'.format(i), concat_layer)


        self.M_base = torch.zeros(input_dim, input_dim).cuda()
        input_dim = input_dim * input_dim


        self.bottleneck_layer = []
        for i,output in enumerate(bottle):
            concat_layer = bottleneck(input_dim, output ,non_linear=self.non_linear)
            input_dim = output
            self.bottleneck_layer.append(concat_layer)
            setattr(self, 'bottle_layer{}'.format(i), concat_layer)

        self.bottle_neck = nn.Linear(input_dim, self.embedding_size)
        self.output_layer = nn.Linear(self.embedding_size, num_classes)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output



    def forward(self,input_x):

        #input data dim. is batch X 1 X feature_size X time
        if not input_x.shape[1] == 1:
            exit(1)
        x= input_x.view((input_x.shape[0],input_x.shape[2],input_x.shape[3]),-1)

        x = x.transpose(1, 2)
        for layer in self.layer[:-1]:
            x = layer(x)
        xb = self.layer[-1](x)

        stack = []
        for b_idx in range(xb.shape[0]):
            M = self.M_base.clone()
            for t_idx in range(xb.shape[2]):
                M = torch.addr(M,x[b_idx,:,t_idx],xb[b_idx,:,t_idx])
            stack.append(M.view(-1))
        x = torch.stack(stack)

        for layer in self.bottleneck_layer:
            x = layer(x)
        x = self.bottle_neck(x)

        self.features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha = 10
        self.features = self.features * alpha

        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.output_layer(features)
        return res


class conv_frame(nn.Module):
    def __init__(self, input_dim,output_dim, use_p,non_linear):
        super(conv_frame,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_p = use_p
        self.kernel_width= use_p[1]
        self.dilation = use_p[2]
        if input_dim == use_p[0] or 1 == use_p[0]:
            if self.kernel_width == 1:
                self.flag = True
                self.conv_h = nn.Linear(input_dim, output_dim)
            else:
                self.flag = False
                self.conv_h = nn.Conv1d(input_dim, output_dim, self.kernel_width,dilation = self.dilation)
        else:
            """this is not implement"""
        self.conv_h_bn = nn.BatchNorm1d(self.output_dim)
        self.non_linear = non_linear


    def forward(self,input_x):
        if self.flag:
            input_x = input_x.transpose(1, 2)
            x = self.conv_h(input_x)
            x = x.transpose(1, 2)
        else:
            x = self.conv_h(input_x)
        x = self.conv_h_bn(x)
        x = self.non_linear(x)

        return x


class bottleneck(nn.Module):
    def __init__(self, input_dim,output_dim, non_linear):
        super(bottleneck,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv_h = nn.Linear(input_dim, output_dim)
        self.conv_h_bn = nn.BatchNorm1d(self.output_dim)
        self.non_linear = non_linear


    def forward(self,input_x):
        x = self.conv_h(input_x)
        x = self.conv_h_bn(x)
        x = self.non_linear(x)
        return x