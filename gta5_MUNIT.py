import os
import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms.functional as F
import torch
import torch.utils.data as data
from PIL import Image
import random
IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_cs_labels(dir):
	images = []
	assert os.path.isdir(dir), '%s is not a valid directory' % dir

	for root, _, fnames in sorted(os.walk(dir)):
		for fname in fnames:
			if is_image_file(fname):
				path = os.path.join(root, fname)
				if path.endswith("_gtFine_labelIds.png"):
					images.append(path)

	return list(set(images))


def make_dataset(dir):
	images = []
	assert os.path.isdir(dir), '%s is not a valid directory' % dir

	for root, _, fnames in sorted(os.walk(dir)):
		for fname in fnames:
			if is_image_file(fname):
				path = os.path.join(root, fname)
				images.append(path)

	return list(set(images))


ignore_label = 255
id2label = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
			3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
			7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
			14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
			18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
			28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
		   220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
		   0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
		   'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
		   'bicycle']


def remap_labels_to_train_ids(arr):
	out = ignore_label * np.ones(arr.shape, dtype=np.uint8)
	for id, label in id2label.items():
		out[arr == id] = int(label)
	return out

class gta(data.Dataset):
	def __init__(self, dataroot, new_size, height, width, train_split=False, crop=False, flip=False):
		self.root = dataroot
		self.train_split = train_split
		self.crop = crop
		self.flip = flip
		self.width = width
		self.height = height

		if train_split:
			self.dir_images = os.path.join(dataroot, 'gta', 'images')
			self.dir_labels = os.path.join(dataroot, 'gta', 'labels')
		else:
			self.dir_images = os.path.join(dataroot, 'gta', 'images')
			self.dir_labels = os.path.join(dataroot, 'gta', 'labels')

		self.path_images = make_dataset(self.dir_images)
		self.path_images = sorted(self.path_images)
		self.size = len(self.path_images)

		self.path_labels = make_dataset(self.dir_labels)
		self.path_labels = sorted(self.path_labels)

		self.transform_1 = get_images_transform_1(new_size)
		self.labels_transform_1 = get_labels_transform_1(new_size)

		self.transform_2 = get_images_transform_2()
		self.labels_transform_2 = get_labels_transform_2()

	def __getitem__(self, index):
		path = self.path_images[index]

		image = Image.open(self.path_images[index]).convert('RGB')

		label = Image.open(self.path_labels[index])
		label = np.asarray(label)
		label = remap_labels_to_train_ids(label)
		label = Image.fromarray(label, 'L')

		# first transform
		image_t = self.transform_1(image)
		label_t = self.labels_transform_1(label)

		# crop
		if self.crop:
			i, j, h, w = transforms.RandomCrop.get_params(image_t, (self.height, self.width))
			image_t = F.crop(image_t, i, j, h, w)
			label_t = F.crop(label_t, i, j, h, w)

		if self.flip:
			if random.random() > 0.5:
				image_t = F.hflip(image_t)
				label_t = F.hflip(label_t)

		# second transform
		image_t = self.transform_2(image_t)
		label_t = self.labels_transform_2(label_t)

		return image_t, label_t, path

	def __len__(self):
		return self.size


def get_images_transform_1(new_size):
	transform_list = []
	transform_list.append(transforms.Lambda(lambda img: __scale_width(img, new_size)))
	return transforms.Compose(transform_list)

def get_labels_transform_1(new_size):
	transform_list = []
	transform_list.append(transforms.Lambda(lambda img: __scale_width_target(img, new_size)))
	return transforms.Compose(transform_list)

def get_images_transform_2():
	transform_list = []
	transform_list += [transforms.ToTensor(),
					   transforms.Normalize((0.5, 0.5, 0.5),
											(0.5, 0.5, 0.5))]
	return transforms.Compose(transform_list)

def get_labels_transform_2():
	transform_list = []
	transform_list.append(transforms.Lambda(lambda img: to_tensor_raw(img)))
	return transforms.Compose(transform_list)


def __scale_width(img, target_width):
	ow, oh = img.size
	if (ow == target_width):
		return img
	w = target_width
	h = int(target_width * oh / ow)
	return img.resize((w, h), Image.BICUBIC)


def __scale_width_target(img, target_width):
	ow, oh = img.size
	if (ow == target_width):
		return img
	w = target_width
	h = int(target_width * oh / ow)
	return img.resize((w, h), Image.NEAREST)


def to_tensor_raw(im):
	return torch.from_numpy(np.array(im, np.int64, copy=False))