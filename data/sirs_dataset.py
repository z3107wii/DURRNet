import math
import os.path
import random
from os.path import join

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

import data.torchdata as torchdata
from data.image_folder import make_dataset
from data.transforms import to_tensor, ReflectionSythesis_1


def __scale_width(img, target_width):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.0) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


def __scale_height(img, target_height):
    ow, oh = img.size
    if oh == target_height:
        return img
    h = target_height
    w = int(target_height * ow / oh)
    w = math.ceil(w / 2.0) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


def paired_data_transforms(img_1, img_2, unaligned_transforms=False):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    target_size = int(random.randint(224, 448) / 2.0) * 2

    ow, oh = img_1.size
    if ow >= oh:
        img_1 = __scale_height(img_1, target_size)
        img_2 = __scale_height(img_2, target_size)
    else:
        img_1 = __scale_width(img_1, target_size)
        img_2 = __scale_width(img_2, target_size)

    if random.random() < 0.5:
        img_1 = F.hflip(img_1)
        img_2 = F.hflip(img_2)

    i, j, h, w = get_params(img_1, (224, 224))
    img_1 = F.crop(img_1, i, j, h, w)

    if unaligned_transforms:
        i_shift = random.randint(-10, 10)
        j_shift = random.randint(-10, 10)
        i += i_shift
        j += j_shift

    img_2 = F.crop(img_2, i, j, h, w)
    return img_1, img_2


def paired_data_transforms_test(img_1, img_2, unaligned_transforms=False):
    def get_params(img, output_size) -> object:
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = 0
        j = 0
        return i, j, th, tw

    target_size = 448

    ow, oh = img_1.size
    if ow >= oh:
        img_1 = __scale_height(img_1, target_size)
        img_2 = __scale_height(img_2, target_size)
    else:
        img_1 = __scale_width(img_1, target_size)
        img_2 = __scale_width(img_2, target_size)

    i, j, h, w = get_params(img_1, (448, 448))
    img_1 = F.crop(img_1, i, j, h, w)

    if unaligned_transforms:
        i_shift = random.randint(-10, 10)
        j_shift = random.randint(-10, 10)
        i += i_shift
        j += j_shift

    img_2 = F.crop(img_2, i, j, h, w)
    return img_1, img_2


def data_transforms(img):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    target_size = int(random.randint(224, 448) / 2.0) * 2
    ow, oh = img.size
    if ow >= oh:
        img = __scale_height(img, target_size)
    else:
        img = __scale_width(img, target_size)

    if random.random() < 0.5:
        img = F.hflip(img)

    i, j, h, w = get_params(img, (224, 224))
    img = F.crop(img, i, j, h, w)
    return img


BaseDataset = torchdata.Dataset


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, *args, **kwargs)
        self.shuffle = shuffle

    def reset(self):
        if self.shuffle:
            print("Reset Dataset...")
            self.dataset.reset()


class CEILDataset(BaseDataset):
    def __init__(
        self,
        datadir,
        fns=None,
        size=None,
        enable_transforms=True,
        low_sigma=2,
        high_sigma=5,
        low_gamma=1.3,
        high_gamma=1.3,
        finetune=False,
    ):
        super(CEILDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.enable_transforms = enable_transforms
        self.finetune = finetune
        sortkey = lambda key: os.path.split(key)[-1]
        self.paths = sorted(make_dataset(datadir, fns), key=sortkey)
        if size is not None:
            self.paths = np.random.choice(self.paths, size)

        self.syn_model = ReflectionSythesis_1(
            kernel_sizes=[11],
            low_sigma=low_sigma,
            high_sigma=high_sigma,
            low_gamma=low_gamma,
            high_gamma=high_gamma,
        )
        self.reset(shuffle=False)

    def reset(self, shuffle=True):
        if shuffle:
            random.shuffle(self.paths)
        num_paths = len(self.paths) // 2
        self.B_paths = self.paths[0:num_paths]
        self.R_paths = self.paths[num_paths : 2 * num_paths]

    def data_synthesis(self, t_img, r_img):
        if self.enable_transforms:
            t_img, r_img = paired_data_transforms(t_img, r_img)
        syn_model = self.syn_model
        t_img, r_img, m_img = syn_model(t_img, r_img)

        B = to_tensor(t_img)
        R = to_tensor(r_img)
        M = to_tensor(m_img)

        return B, R, M

    def __getitem__(self, index):
        index_B = index % len(self.B_paths)
        index_R = index % len(self.R_paths)

        B_path = self.B_paths[index_B]
        R_path = self.R_paths[index_R]

        t_img = Image.open(B_path).convert("RGB")
        r_img = Image.open(R_path).convert("RGB")

        if self.finetune:
            B = to_tensor(t_img)
            R = to_tensor(Image.fromarray(np.zeros_like(t_img)))
            M = to_tensor(t_img)
        else:
            B, R, M = self.data_synthesis(t_img, r_img)

        fn = os.path.basename(B_path)
        return {
            "input": M,
            "target_t": B,
            "target_r": R,
            "fn": fn,
            "identity": self.finetune,
            "identity_r": False,
        }

    def __len__(self):
        if self.size is not None:
            return min(max(len(self.B_paths), len(self.R_paths)), self.size)
        else:
            return max(len(self.B_paths), len(self.R_paths))


class CEILTrainDataset(BaseDataset):
    def __init__(
        self,
        datadir,
        fns=None,
        size=None,
        enable_transforms=False,
        unaligned_transforms=False,
        round_factor=1,
        flag=None,
        clip=None,
        finetune=False,
        if_align=False,
        m_dir="blended",
        t_dir="transmission_layer",
    ):
        super(CEILTrainDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.m_dir = m_dir
        self.t_dir = t_dir
        self.fns = fns or os.listdir(join(datadir, m_dir))
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        self.finetune = finetune
        self.if_align = if_align

        if size is not None:
            self.fns = self.fns[:size]

    def align(self, x1, x2):
        h, w = x1.height, x1.width
        h, w = h // 16 * 16, w // 16 * 16
        x1 = x1.resize((w, h))
        x2 = x2.resize((w, h))
        return x1, x2

    def __getitem__(self, index):
        fn = self.fns[index]
        t_img = Image.open(join(self.datadir, self.t_dir, fn)).convert("RGB")
        m_img = Image.open(join(self.datadir, self.m_dir, fn)).convert("RGB")

        if self.if_align:
            t_img, m_img = self.align(t_img, m_img)
        if self.enable_transforms:
            t_img, m_img = paired_data_transforms(
                t_img, m_img, self.unaligned_transforms
            )

        B = to_tensor(t_img)
        M = to_tensor(t_img) if self.finetune else to_tensor(m_img)

        dic = {
            "input": M,
            "target_t": B,
            "fn": fn,
            "real": True,
            "target_r": M - B,
            "identity": self.finetune,
            "identity_r": False,
        }
        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        return min(len(self.fns), self.size) if self.size is not None else len(self.fns)


class CEILTestDataset(BaseDataset):
    def __init__(
        self,
        datadir,
        fns=None,
        size=None,
        enable_transforms=False,
        unaligned_transforms=False,
        round_factor=1,
        flag=None,
        clip=None,
        finetune=False,
        if_align=False,
        m_dir="blended",
        t_dir="transmission_layer",
    ):
        super(CEILTestDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.m_dir = m_dir
        self.t_dir = t_dir
        self.fns = fns or os.listdir(join(datadir, m_dir))
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        self.finetune = finetune
        self.if_align = if_align

        if size is not None:
            self.fns = self.fns[:size]

    def align(self, x1, x2):
        h, w = x1.height, x1.width
        h, w = h // 16 * 16, w // 16 * 16
        x1 = x1.resize((w, h))
        x2 = x2.resize((w, h))
        return x1, x2

    def __getitem__(self, index):
        fn = self.fns[index]
        t_img = Image.open(join(self.datadir, self.t_dir, fn)).convert("RGB")
        m_img = Image.open(join(self.datadir, self.m_dir, fn)).convert("RGB")

        if self.if_align:
            t_img, m_img = self.align(t_img, m_img)
        if self.enable_transforms:
            t_img, m_img = paired_data_transforms_test(
                t_img, m_img, self.unaligned_transforms
            )

        B = to_tensor(t_img)
        M = to_tensor(t_img) if self.finetune else to_tensor(m_img)

        dic = {
            "input": M,
            "target_t": B,
            "fn": fn,
            "real": True,
            "target_r": M - B,
            "identity": self.finetune,
            "identity_r": False,
        }
        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        return min(len(self.fns), self.size) if self.size is not None else len(self.fns)


class RealDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None):
        super(RealDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir))
        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]
        m_img = Image.open(join(self.datadir, fn)).convert("RGB")
        M = to_tensor(m_img)
        return {"input": M, "target_t": -1, "fn": fn}

    def __len__(self):
        return min(len(self.fns), self.size) if self.size is not None else len(self.fns)


class FusionDataset(BaseDataset):
    def __init__(self, datasets, fusion_ratios=None):
        self.datasets = datasets
        self.size = sum([len(dataset) for dataset in datasets])
        self.fusion_ratios = fusion_ratios or [1.0 / len(datasets)] * len(datasets)
        print(
            "[i] using a fusion dataset: %d %s imgs fused with ratio %s"
            % (self.size, [len(dataset) for dataset in datasets], self.fusion_ratios)
        )

    def reset(self):
        for dataset in self.datasets:
            dataset.reset()

    def __getitem__(self, index):
        residual = 1
        for i, ratio in enumerate(self.fusion_ratios):
            if random.random() < ratio / residual or i == len(self.fusion_ratios) - 1:
                dataset = self.datasets[i]
                return dataset[index % len(dataset)]
            residual -= ratio

    def __len__(self):
        return self.size
