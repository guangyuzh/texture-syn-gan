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

### four layers PSGAN
class PSGAN(object):
    class _netG(nn.Module):
        def __init__(self, ngpu, nz, ngf, nc):
            super().__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                # dim(Z) = nz, 5, 5
                nn.ConvTranspose2d(nz, ngf * 4, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 9 x 9
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 17 x 17
                nn.ConvTranspose2d(ngf * 2, ngf, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 33 x 33
                nn.ConvTranspose2d(ngf, nc, 5, 2, 2, bias=False),
                nn.Tanh()
                # state size. (nc) x 65 x 65
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
                # input is (nc) x 65 x 65
                nn.Conv2d(nc, ndf, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 33 x 33
                nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 17 x 17
                nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 9 x 9
                nn.Conv2d(ndf * 4, 1, 5, 2, 2, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)

            return output.view(-1, 1).squeeze(1)
