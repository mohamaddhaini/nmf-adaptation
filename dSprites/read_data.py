
"""Utility helpers for loading dSprites data for domain adaptation experiments."""
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from PIL import Image



class TextData():
    """Iterable dataset that yields mixed source/target batches with labels."""

    def __init__(self, text_file, label_file, source_batch_size=64, target_batch_size=64, val_batch_size=4):
        """Load the serialized arrays and initialize the batch sampling state."""
        all_text = np.load(text_file)
        self.source_text = all_text[0:92664, :]
        self.target_text = all_text[92664:, :]
        self.val_text = all_text[0:92664, :]
        all_label = np.load(label_file)
        self.label_source = all_label[0:92664, :]
        self.label_target = all_label[92664:, :]
        self.label_val = all_label[0:92664, :]
        self.scaler = StandardScaler().fit(all_text)
        self.source_id = 0
        self.target_id = 0
        self.val_id = 0
        self.source_size = self.source_text.shape[0]
        self.target_size = self.target_text.shape[0]
        self.val_size = self.val_text.shape[0]
        self.source_batch_size = source_batch_size
        self.target_batch_size = target_batch_size
        self.val_batch_size = val_batch_size
        self.source_list = random.sample(range(self.source_size), self.source_size)
        self.target_list = random.sample(range(self.target_size), self.target_size)
        self.val_list = random.sample(range(self.val_size), self.val_size)
        self.feature_dim = self.source_text.shape[1]

    def next_batch(self, train=True):
        """Return the next batch of standardized features and labels."""
        data = []
        label = []
        if train:
            remaining = self.source_size - self.source_id
            start = self.source_id
            if remaining <= self.source_batch_size:
                for i in self.source_list[start:]:
                    data.append(self.source_text[i, :])
                    label.append(self.label_source[i, :])
                    self.source_id += 1
                self.source_list = random.sample(range(self.source_size), self.source_size)
                self.source_id = 0
                for i in self.source_list[0:(self.source_batch_size - remaining)]:
                    data.append(self.source_text[i, :])
                    label.append(self.label_source[i, :])
                    self.source_id += 1
            else:
                for i in self.source_list[start:start + self.source_batch_size]:
                    data.append(self.source_text[i, :])
                    label.append(self.label_source[i, :])
                    self.source_id += 1
            remaining = self.target_size - self.target_id
            start = self.target_id
            if remaining <= self.target_batch_size:
                for i in self.target_list[start:]:
                    data.append(self.target_text[i, :])
                    self.target_id += 1
                self.target_list = random.sample(range(self.target_size), self.target_size)
                self.target_id = 0
                for i in self.target_list[0:self.target_batch_size - remaining]:
                    data.append(self.target_text[i, :])
                    self.target_id += 1
            else:
                for i in self.target_list[start:start + self.target_batch_size]:
                    data.append(self.target_text[i, :])
                    self.target_id += 1
        else:
            remaining = self.val_size - self.val_id
            start = self.val_id
            if remaining <= self.val_batch_size:
                for i in self.val_list[start:]:
                    data.append(self.val_text[i, :])
                    label.append(self.label_val[i, :])
                    self.val_id += 1
                self.val_list = random.sample(range(self.val_size), self.val_size)
                self.val_id = 0
                for i in self.val_list[0:self.val_batch_size - remaining]:
                    data.append(self.val_text[i, :])
                    label.append(self.label_val[i, :])
                    self.val_id += 1
            else:
                for i in self.val_list[start:start + self.val_batch_size]:
                    data.append(self.val_text[i, :])
                    label.append(self.label_val[i, :])
                    self.val_id += 1
        data = self.scaler.transform(np.vstack(data))
        label = np.vstack(label)
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()


def make_dataset(image_list, labels):
    """Create (path, label) tuples with discrete integer targets."""
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in xrange(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def make_dataset_r(image_list, labels):
    """Create (path, label) tuples with regression targets."""
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in xrange(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([float(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], float(val.split()[1])) for val in image_list]
    return images

def pil_loader(path):
    """Load an RGB PIL image from disk."""
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def pil_loader1(path):
    """Load a grayscale PIL image from disk."""
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

def accimage_loader(path):
    """Attempt to load images with accimage, falling back to PIL if needed."""
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
    """Default RGB loader that currently wraps PIL."""
    return pil_loader(path)

def default_loader1(path):
    """Default grayscale loader that currently wraps PIL."""
    return pil_loader1(path)

class ImageList(object):
    """Simple image dataset abstraction holding file paths and discrete labels."""
    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader):
        """Store the file/label pairs plus the transforms to apply."""
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise RuntimeError("Found 0 images. Please provide at least one labeled path.")

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """Return the transformed image and target for the provided index."""
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Length of the source dataset."""
        return len(self.imgs)

class ImageList_r(object):
    """Image dataset variant that expects regression labels."""

    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader):
        """Store the file/label pairs plus the transforms to apply."""
        imgs = make_dataset_r(image_list, labels)
        if len(imgs) == 0:
            raise RuntimeError("Found 0 images. Please provide at least one labeled path.")

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """Return the transformed image and regression target for the provided index."""
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Length of the regression dataset."""
        return len(self.imgs)


def ClassSamplingImageList(image_list, transform, return_keys=False):
    """Group image paths by class id and wrap each group in an ImageList."""
    data = open(image_list).readlines()
    label_dict = {}
    for line in data:
        label_dict[int(line.split()[1])] = []
    for line in data:
        label_dict[int(line.split()[1])].append(line)
    all_image_list = {}
    for i in label_dict.keys():
        all_image_list[i] = ImageList(label_dict[i], transform=transform)
    if return_keys:
        return all_image_list, label_dict.keys()
    else:
        return all_image_list
