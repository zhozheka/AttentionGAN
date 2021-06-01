## python train.py --dataset_mode cropped --dataroot  xxx  --model attention_gan --niter 40 --niter_decay 10  --name tooth_consistensy_0.1_cropped_highres___ --batch_size 2

import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
from glob import glob
from torchvision import transforms


def random(key):
    np.random.seed(hash(key) % 2**32)
    r = np.random.rand()
    return r


class TeethDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        # tooth_root = '/home/e-zholkovskiy/data/ffhq'
        # self.A_paths = []
        # with open(os.path.join(tooth_root, 'tooth_probs.txt')) as f:
        #     for line in f:
        #         name, prob = line.strip().split(';')
        #         if float(prob) > 0.85:
        #             self.A_paths.append(
        #                 os.path.join(tooth_root, 'cropped', name))


        tooth_root = '/home/e-zholkovskiy/data/ffhq'
        self.A_paths = []
        with open(os.path.join(tooth_root, 'ffhq_teeth.txt')) as f:
            for line in f:
                name = line.strip()
                path = os.path.join(tooth_root, self.ffhq_dir, name)
                if os.path.isfile(path):
                    self.A_paths.append(path)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(glob('/home/e-zholkovskiy/data/ya_braces/{}/*'.format(self.braces_dir)))

        val_paths = [p for p in self.B_paths if random(os.path.basename(p)) <= 0.1]
        self.B_paths = [p for p in self.B_paths if random(os.path.basename(p)) > 0.1]

        print('\n'.join(val_paths))
        print('len self.A_paths: {}'.format(len(self.A_paths)))
        print('len self.B_paths: {}'.format(len(self.B_paths)))

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        # self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        # self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        # h = 128
        # w = 256
        trs = transforms.Compose([
            transforms.Resize((int(1.1 * self.h), int(1.1 * self.w))),
            transforms.RandomCrop((self.h, self.w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform_A = self.transform_B = trs

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = np.random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


