from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class SGAN(object):
    class _netG(nn.Module):
        def __init__(self, ngpu, nz, ngf, nc):
            super().__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is nz x 5 x 5
                # output is nc x 127 x 127
                nn.ConvTranspose2d(nz, ngf * 8, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 4, ngf * 2, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 2, ngf, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf, nc, 5, 2, 2, bias=False),
                nn.Tanh()
            )

        def forward(self, input):
            if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)
            return output


    class _netD(nn.Module):
        def __init__(self, ngpu, nc, ndf):
            super().__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is nc x 127 x 127
                # output is 1 x 5 x 5
                nn.Conv2d(nc, ndf, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4, ndf * 8, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 8, 1, 5, 2, 2, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)

            return output.view(-1, 1).squeeze(1)

### five layers PSGAN
class PSGAN(object):
    class _netG(nn.Module):
        def __init__(self, ngpu, nz, ngf, nc):
            super().__init__()
            self.ngpu = ngpu
            self.nh = 60
            self.nz_global = 10
            self.nz_period = 2
            self.nw = 5
            self.main = nn.Sequential(
                # input is nz x 5 x 5
                # output is nc x 127 x 127
                nn.ConvTranspose2d(nz, ngf * 8, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 4, ngf * 2, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 2, ngf, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf, nc, 5, 2, 2, bias=False),
                nn.Tanh()
            )
            self.shared_linear = nn.Linear(nz_global, nh)
            self.aux1 = nn.Sequential(
                self.shared_linear,
                nn.ReLU(True),
                nn.Linear(nh, nz_period)
            )

            self.aux2 = nn.Sequential(
                self.shared_linear,
                nn.ReLU(True),
                nn.Linear(nh, nz_period)
            )

        def forward(self, l_tensor, g_tensor, phi):
            # l_tensor : batch_size x nz_local x nw x nw
            # g_tensor : batch_size x nz_global x 1 x 1
            # p_tensor : batch_size x nz_period x nw x nw
            
            # infer batch_size and nw from input
            batch_size, _, nw, _ = l_tensor.size()

            g_tensor_expanded = g_tensor.repeat(1, 1, nw, nw)
            k1 = self.aux1(g_tensor.squeeze())
            k2 = self.aux2(g_tensor.squeeze())
            xx = torch.arrange(self.nw).repeat(nw, 1).repeat(batch_size, self.nz_period)
            yy = torch.arrange(self.nw).repeat(nw, 1).t().repeat(batch_size, self.nz_period)
            p_tensor = torch.sin(k1*xx + k2*yy + phi)

            input = torch.cat([l_tensor, g_tensor_expanded, p_tensor], 1)
            # output : batch_size x nz x nw x nw
            # nz = nz_local + nz_global + nz_period
            output = self.main(input)
            return output


    class _netD(nn.Module):
        def __init__(self, ngpu, nc, ndf):
            super().__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is nc x 127 x 127
                # output is 1 x 5 x 5
                nn.Conv2d(nc, ndf, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4, ndf * 8, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 8, 1, 5, 2, 2, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)

            return output.view(-1, 1).squeeze(1)
