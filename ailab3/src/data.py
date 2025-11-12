import os
import re
from typing import List, Tuple, Dict, Callable, Set

import torch
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from PIL import Image


IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def _is_letter_class(name: str) -> bool:
    return bool(re.fullmatch(r"[A-HJ-NP-Z]", name))


def _is_chinese_char(name: str) -> bool:
    return any(0x4E00 <= ord(ch) <= 0x9FFF for ch in name)


def _list_classes(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    return sorted([d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))])


def _build_transform(img_size: int = 28) -> transforms.Compose:
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

def _build_train_transform(img_size: int = 28, augment: bool = False) -> transforms.Compose:
    if not augment:
        return _build_transform(img_size)
    # ???????????????/???/????????????????????
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.95, 1.05), shear=5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


class FilteredImageFolder(Dataset):
    def __init__(self, root: str, classes: List[str], class_to_idx: Dict[str, int] | None = None, transform=None):
        self.root = root
        self.classes = sorted(classes)
        if class_to_idx is None:
            class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.class_to_idx = class_to_idx
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root, cls)
            if not os.path.isdir(cls_dir):
                continue
            label = self.class_to_idx[cls]
            for dirpath, _, filenames in os.walk(cls_dir):
                for fname in filenames:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in IMG_EXTENSIONS:
                        self.samples.append((os.path.join(dirpath, fname), label))

        self.targets = [y for _, y in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as img:
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_digits_letters_datasets(base_dir: str, img_size: int = 28, batch_size: int = 64, augment: bool = False) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    num_train = os.path.join(base_dir, 'dataset_num', 'train_28_28')
    num_test = os.path.join(base_dir, 'dataset_num', 'testset_28_28')

    tfm_train = _build_train_transform(img_size, augment=augment)
    tfm_test = _build_transform(img_size)

    digits_classes = [str(i) for i in range(10) if os.path.isdir(os.path.join(num_train, str(i)))]
    # ?????????? dataset_num??train ?? test ????????????????
    union_letter_dirs = sorted(set(_list_classes(num_train)) | set(_list_classes(num_test)))
    letter_classes = [c for c in union_letter_dirs if _is_letter_class(c)]

    letters_idx = {c: i for i, c in enumerate(sorted(letter_classes))}
    digits_idx = {c: i + len(letters_idx) for i, c in enumerate(sorted(digits_classes))}

    # ?????????? dataset_num
    train_letters = FilteredImageFolder(num_train, letter_classes, class_to_idx=letters_idx, transform=tfm_train)
    test_letters = FilteredImageFolder(num_test, letter_classes, class_to_idx=letters_idx, transform=tfm_test)
    train_digits = FilteredImageFolder(num_train, digits_classes, class_to_idx=digits_idx, transform=tfm_train)
    test_digits = FilteredImageFolder(num_test, digits_classes, class_to_idx=digits_idx, transform=tfm_test)

    train_ds = ConcatDataset([train_letters, train_digits])
    test_ds = ConcatDataset([test_letters, test_digits])

    num_classes = len(letters_idx) + len(digits_idx)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, num_classes


def get_chinese_datasets(base_dir: str, img_size: int = 28, batch_size: int = 64, augment: bool = False) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    char_train = os.path.join(base_dir, 'dataset_char', 'train_28_28')
    char_test = os.path.join(base_dir, 'dataset_char', 'test_28_28')

    tfm_train = _build_train_transform(img_size, augment=augment)
    tfm_test = _build_transform(img_size)

    union_classes = sorted(set(_list_classes(char_train)) | set(_list_classes(char_test)))
    chinese_classes = [c for c in union_classes if _is_chinese_char(c)]

    chinese_idx = {c: i for i, c in enumerate(sorted(chinese_classes))}

    train_ds = FilteredImageFolder(char_train, chinese_classes, class_to_idx=chinese_idx, transform=tfm_train)
    test_ds = FilteredImageFolder(char_test, chinese_classes, class_to_idx=chinese_idx, transform=tfm_test)

    num_classes = len(chinese_idx)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, num_classes