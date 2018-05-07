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
from prediction import PredOpt


class GANNetwork(object):

    class RandomImageFolder(dset.ImageFolder):
        def __init__(self, *args, **kwargs):
            self.n_sample  = kwargs['n_sample']
            self.npx       = kwargs['npx']
            self.imageSize = kwargs['imageSize']
            kwargs.pop('n_sample')
            kwargs.pop('npx')
            kwargs.pop('imageSize')
            super().__init__(*args, **kwargs)

            self.offsets = []
            for i in range(self.n_sample):
                ox = random.randint(0, self.imageSize - self.npx)
                oy = random.randint(0, self.imageSize - self.npx)
                self.offsets.append((ox, oy))

        def __len__(self):
            return super().__len__() * self.n_sample

        def __getitem__(self, ix):
            img_ix, sample_ix = divmod(ix, self.n_sample)
            img, label = super().__getitem__(img_ix)
            ox, oy = self.offsets[sample_ix]
            img = img[:, ox:ox+self.npx, oy:oy+self.npx]
            return img, label

    def __init__(self, opt):
        self.opt = opt
        self._rand_seed()

        cudnn.benchmark = True
        if torch.cuda.is_available() and not self.opt.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        self._load_dataset()

        self.ngpu     = int(self.opt.ngpu)
        self.nz       = int(self.opt.nz)
        self.ngf      = int(self.opt.ngf)
        self.ndf      = int(self.opt.ndf)
        self.n_sample = int(self.opt.n_sample)
        self.npx      = int(self.opt.npx)
        self.nw       = int(self.opt.nw)
        self.ntw      = int(self.opt.ntw)
        self.nc       = 3
        self.pred     = self.opt.pred

        self._init_netG()
        self._init_netD()
        self.criterion = nn.BCELoss()
        self._init_input()

        # setup optimizer
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        # Prediction in https://openreview.net/forum?id=Skj8Kag0Z&noteId=rkLymJTSf
        self.optimizer_predD = PredOpt(self.netD.parameters())
        self.optimizer_predG = PredOpt(self.netG.parameters())
        self.lookahead_step = 1.0 if self.pred else 0.0

    def _rand_seed(self):
        if self.opt.manualSeed is None:
            self.opt.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", self.opt.manualSeed)
        random.seed(self.opt.manualSeed)
        torch.manual_seed(self.opt.manualSeed)
        if self.opt.cuda:
            torch.cuda.manual_seed_all(self.opt.manualSeed)

    def _load_dataset(self):
        dataset = self.RandomImageFolder(root=self.opt.dataroot,
            transform=transforms.Compose([
                transforms.Resize(self.opt.imageSize),
                transforms.CenterCrop(self.opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            imageSize=self.opt.imageSize,
            npx=self.opt.npx,
            n_sample=self.opt.n_sample
        )
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.opt.batchSize, shuffle=True, num_workers=int(self.opt.workers))

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
        self.input = torch.FloatTensor(self.opt.batchSize, self.nc, self.npx, self.npx)
        self.noise = torch.FloatTensor(self.opt.batchSize, self.nz, self.nw, self.nw)
        self.fixed_noise = torch.FloatTensor(self.opt.batchSize, self.nz, self.nw, self.nw).normal_(0, 1)
        self.label = torch.FloatTensor(self.opt.batchSize, self.nw, self.nw)
        self.real_label = 1
        self.fake_label = 0

        if self.opt.cuda:
            self.netD.cuda()
            self.netG.cuda()
            self.criterion.cuda()
            self.input, self.label = self.input.cuda(), self.label.cuda()
            self.noise, self.fixed_noise = self.noise.cuda(), self.fixed_noise.cuda()
        self.fixed_noise = Variable(self.fixed_noise)

    def test(self, input_sz, epoch):
        ### generate texture using netG
        ### taking input of size 1 * zdim * input_sz * input_sz
        noise = torch.FloatTensor(1, self.nz, input_sz, input_sz)
        if self.opt.cuda:
            noise = noise.cuda()
        noise.normal_(0, 1)
        noise = Variable(noise, volatile=True)
        fake = self.netG(noise)
        vutils.save_image(fake.data,
            '%s/texture_output_%03d.png' % (self.opt.outf, epoch),
            normalize=True
        )

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
                batch_size = real_cpu.size(0)
                if self.opt.cuda:
                    real_cpu = real_cpu.cuda()
                self.input.resize_as_(real_cpu).copy_(real_cpu)
                self.label.resize_(batch_size * self.nw**2).fill_(self.real_label)
                inputv = Variable(self.input)
                labelv = Variable(self.label)

                output = self.netD(inputv)
                errD_real = self.criterion(output, labelv)
                errD_real.backward()
                D_x = output.data.mean()

                # train with fake
                self.noise.resize_(batch_size, self.nz, self.nw, self.nw).normal_(0, 1)
                noisev = Variable(self.noise)

                # Compute gradient of D w/ predicted G
                with self.optimizer_predG.lookahead(step=self.lookahead_step):
                    fake = self.netG(noisev)
                    labelv = Variable(self.label.fill_(self.fake_label))
                    output = self.netD(fake.detach())
                    errD_fake = self.criterion(output, labelv)
                    errD_fake.backward()
                    D_G_z1 = output.data.mean()
                    errD = errD_real + errD_fake
                    self.optimizerD.step()
                    self.optimizer_predD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                labelv = Variable(self.label.fill_(self.real_label))  # fake labels are real for generator cost

                # Compute gradient of G w/ predicted D
                with self.optimizer_predD.lookahead(step=self.lookahead_step):
                    fake = self.netG(noisev)
                    output = self.netD(fake)
                    errG = self.criterion(output, labelv)
                    errG.backward()
                    D_G_z2 = output.data.mean()
                    self.optimizerG.step()
                    self.optimizer_predG.step()

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
            self.test(self.ntw, epoch)

            # do checkpointing
            torch.save(self.netG.state_dict(), '%s/netG_epoch_%d.pth' % (self.opt.outf, epoch))
            torch.save(self.netD.state_dict(), '%s/netD_epoch_%d.pth' % (self.opt.outf, epoch))
