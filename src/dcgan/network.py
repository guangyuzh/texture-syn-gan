import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from gan import _netG, _netD, weights_init


class GANNetwork(object):

    def __init__(self, opt):
        self.opt = opt
        self._rand_seed()

        cudnn.benchmark = True
        if torch.cuda.is_available() and not self.opt.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        self._load_dataset()

        self.ngpu = int(self.opt.ngpu)
        self.nz = int(self.opt.nz)
        self.ngf = int(self.opt.ngf)
        self.ndf = int(self.opt.ndf)
        self.n_sample = int(self.opt.n_sample)
        self.npx = int(self.opt.npx)
        self.nc = 3
        self._init_netG()
        self._init_netD()
        self.criterion = nn.BCELoss()
        self._init_input()

        # setup optimizer
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def _rand_seed(self):
        if self.opt.manualSeed is None:
            self.opt.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", self.opt.manualSeed)
        random.seed(self.opt.manualSeed)
        torch.manual_seed(self.opt.manualSeed)
        if self.opt.cuda:
            torch.cuda.manual_seed_all(self.opt.manualSeed)

    def _load_dataset(self):
        if self.opt.dataset in ['imagenet', 'folder', 'lfw']:
            # folder dataset
            dataset = dset.ImageFolder(root=self.opt.dataroot,
                                       transform=transforms.Compose([
                                           transforms.Resize(self.opt.imageSize),
                                           transforms.CenterCrop(self.opt.imageSize),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))
        elif self.opt.dataset == 'lsun':
            dataset = dset.LSUN(self.opt.dataroot, classes=['bedroom_val'],
                                transform=transforms.Compose([
                                    transforms.Resize(self.opt.imageSize),
                                    transforms.CenterCrop(self.opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        elif self.opt.dataset == 'cifar10':
            dataset = dset.CIFAR10(root=self.opt.dataroot, download=True,
                                   transform=transforms.Compose([
                                       transforms.Resize(self.opt.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        elif self.opt.dataset == 'fake':
            dataset = dset.FakeData(image_size=(3, self.opt.imageSize, self.opt.imageSize),
                                    transform=transforms.ToTensor())
        assert dataset
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.opt.batchSize,
                                                    shuffle=True, num_workers=int(self.opt.workers))

    def _init_netG(self):
        self.netG = _netG(self.ngpu, self.nz, self.ngf, self.nc)
        self.netG.apply(weights_init)
        if self.opt.netG != '':
            self.netG.load_state_dict(torch.load(self.opt.netG))
        print(self.netG)

    def _init_netD(self):
        self.netD = _netD(self.ngpu, self.nc, self.ndf)
        self.netD.apply(weights_init)
        if self.opt.netD != '':
            self.netD.load_state_dict(torch.load(self.opt.netD))
        print(self.netD)

    def _init_input(self):
        self.input = torch.FloatTensor(self.opt.batchSize, self.nc, self.opt.imageSize, self.opt.imageSize)
        self.noise = torch.FloatTensor(self.opt.batchSize, self.nz, 1, 1)
        self.fixed_noise = torch.FloatTensor(self.opt.batchSize, self.nz, 1, 1).normal_(0, 1)
        self.label = torch.FloatTensor(self.opt.batchSize)
        self.real_label = 1
        self.fake_label = 0

        if self.opt.cuda:
            self.netD.cuda()
            self.netG.cuda()
            self.criterion.cuda()
            self.input, self.label = self.input.cuda(), self.label.cuda()
            self.noise, self.fixed_noise = self.noise.cuda(), self.fixed_noise.cuda()
        self.fixed_noise = Variable(self.fixed_noise)

    def _truncate(self, data):
        sub_samples = []
        for batch in data:
            _l, _m = batch.shape[1], batch.shape[2]
            assert _l == _m
            for i in range(self.n_sample):
                idx_x = random.randint(0, _l - self.npx)
                idx_y = random.randint(0, _m - self.npx)
                sub_sample = batch[:, idx_x:idx_x+self.npx, idx_y:idx_y+self.npx]
                sub_samples.append(sub_sample)
        return torch.stack(sub_samples)

    def train(self):
        cudnn.benchmark = True
        for epoch in range(self.opt.niter):
            for i, data in enumerate(self.dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                self.netD.zero_grad()
                real_cpu, _ = data
                real_cpu = self._truncate(real_cpu)
                batch_size = real_cpu.size(0)
                if self.opt.cuda:
                    real_cpu = real_cpu.cuda()
                self.input.resize_as_(real_cpu).copy_(real_cpu)
                self.label.resize_(batch_size).fill_(self.real_label)
                inputv = Variable(self.input)
                labelv = Variable(self.label)

                output = self.netD(inputv)
                errD_real = self.criterion(output, labelv)
                errD_real.backward()
                D_x = output.data.mean()

                # train with fake
                self.noise.resize_(batch_size, self.nz, 1, 1).normal_(0, 1)
                noisev = Variable(self.noise)
                fake = self.netG(noisev)
                labelv = Variable(self.label.fill_(self.fake_label))
                output = self.netD(fake.detach())
                errD_fake = self.criterion(output, labelv)
                errD_fake.backward()
                D_G_z1 = output.data.mean()
                errD = errD_real + errD_fake
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                labelv = Variable(self.label.fill_(self.real_label))  # fake labels are real for generator cost
                output = self.netD(fake)
                errG = self.criterion(output, labelv)
                errG.backward()
                D_G_z2 = output.data.mean()
                self.optimizerG.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, self.opt.niter, i, len(self.dataloader),
                         errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
                if i % 100 == 0:
                    vutils.save_image(real_cpu,
                                      '%s/real_samples.png' % self.opt.outf,
                                      normalize=True)
                    fake = self.netG(self.fixed_noise)
                    vutils.save_image(fake.data,
                                      '%s/fake_samples_epoch_%03d.png' % (self.opt.outf, epoch),
                                      normalize=True)

            # do checkpointing
            torch.save(self.netG.state_dict(), '%s/netG_epoch_%d.pth' % (self.opt.outf, epoch))
            torch.save(self.netD.state_dict(), '%s/netD_epoch_%d.pth' % (self.opt.outf, epoch))
