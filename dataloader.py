import numpy as np
import torch
from torch.utils.data import Dataset
import random, PIL
from PIL import Image, ImageDraw
from torchvision import datasets, transforms


class MovingObjects(Dataset):
    def __init__(self, mode, transform, my_seed, dummy_len=30000):
        print("NOTE: no data normalization and data range is [0,1]")
        if mode is "train":
            random.seed(my_seed)
            torch.manual_seed(my_seed)
            np.random.seed(my_seed)
        #else:
         #   random.seed(1234)
          #  torch.manual_seed(1234)
           # np.random.seed(1234)

        # constant speed of 4 pixels
        self.seq_len = 3
        # 8 possible direction of movement
        pix = 8
        self.dummy_len = dummy_len
        self.deltaxy = [(pix,pix), (pix,0), (pix,-pix), (0,pix), (0,-pix), (-pix,-pix), (-pix,0), (-pix,pix)]
        self.shapes = ['circle', 'rectangle', 'polygon']
        self.size_range =  [18] #, 26] # range(10, 20)

        self.center_xy = np.array([32, 32])
        self.transform = transform

    def __len__(self):
        return self.dummy_len

    def __getitem__(self, idx):
        """idx is a dummy value"""
        # pick a shape
        shape = random.choice(self.shapes)

        # pick a direction
        deltax, deltay = random.choice(self.deltaxy)

        # pick size
        size = random.choice(self.size_range)

        # pick color
        r = random.choice(range(0,256,200))
        g = random.choice(range(0,256,200))
        b = random.choice(range(0,256,200))

        frames = []
        img1 = PIL.Image.new(mode='RGB', size=(64,64), color='gray')
        if shape is 'circle':
            for i in range(self.seq_len):

                c_img = img1.copy()
                c_draw = ImageDraw.Draw(c_img)
                c1 = tuple(self.center_xy - size/2 + np.array([deltax, deltay])*i)
                c2 = tuple(self.center_xy + size/2 + np.array([deltax, deltay])*i)

                c_draw.ellipse([c1, c2], fill=(r ,g , b))
                frames.append(self.transform(c_img))

        elif shape is 'rectangle':
            for i in range(self.seq_len):

                c_img = img1.copy()
                c_draw = ImageDraw.Draw(c_img)
                c1 = tuple(self.center_xy - size/2 + np.array([deltax, deltay])*i)
                c2 = tuple(self.center_xy + size/2 + np.array([deltax, deltay])*i)

                c_draw.rectangle([c1, c2], fill=(r ,g , b))
                frames.append(self.transform(c_img))

        elif shape is 'polygon':
            for i in range(self.seq_len):
                c_img = img1.copy()
                c_draw = ImageDraw.Draw(c_img)

                c1 = tuple(self.center_xy - np.array([0, size/2]) + np.array([deltax, deltay])*i)
                c2 = tuple(self.center_xy + np.array([size/2, size/3]) + np.array([deltax, deltay])*i)
                c3 = tuple(self.center_xy + np.array([-size/2, size/3]) + np.array([deltax, deltay])*i)

                c_draw.polygon([c1, c2, c3], fill=(r ,g , b))

                #change to tensor
                frames.append(self.transform(c_img))

        else:
            raise NotImplementedError()

        frames_tensor = torch.stack(frames, dim=0)
        return frames_tensor


class MovingMNIST(object):

    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, seq_len=20, num_digits=2, image_size=64, deterministic=True):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = 32
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.channels = 1

        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Scale(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size,
                      image_size,
                      self.channels),
                    dtype=np.float32)
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, 5)
                        dx = np.random.randint(-4, 5)
                elif sy >= image_size-32:
                    sy = image_size-32-1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-4, 0)
                        dx = np.random.randint(-4, 5)

                if sx < 0:
                    sx = 0
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, 5)
                        dy = np.random.randint(-4, 5)
                elif sx >= image_size-32:
                    sx = image_size-32-1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-4, 0)
                        dy = np.random.randint(-4, 5)

                x[t, sy:sy+32, sx:sx+32, 0] += digit.numpy().squeeze()
                sy += dy
                sx += dx

        x[x>1] = 1.
        # t, w, h, c --> t, c, w, h
        return x.transpose([0,3,1,2])
