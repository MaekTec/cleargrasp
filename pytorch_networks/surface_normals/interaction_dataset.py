import os
from matplotlib.pyplot import imread

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import glob
import itertools
from collections import defaultdict
import pickle
from transforms import ToFloatImageFromUint8Image
from torchvision.transforms import transforms
from imgaug import augmenters as iaa
import imgaug as ia


class InteractionDataset(Dataset):

    def __init__(
            self,
            data_folder="dataset",
            transform=None,
            target_transform=None,
            train: bool = False,
            val: bool = False,
            test: bool = False,
            split: tuple = (0.7, 0.2, 0.1)  # (train, val, test)
    ):
        super(InteractionDataset, self).__init__()
        self.data_folder = data_folder
        self.transform = transform
        self.target_transform = target_transform
        assert sum([train, val, test]) == 1  # only one can be active
        assert any([train, val, test])  # at least one hast to active
        assert np.isclose(sum(split), 1.0)

        # for whole dataset
        paths = list(glob.iglob(os.path.join(data_folder, '20**/**/'), recursive=True))[1:]
        episode_to_file_paths = defaultdict(lambda: [])
        for path in paths:
            files = glob.glob(os.path.join(path, "*.pickle"))
            files.sort(key=lambda x: x.split(os.sep)[-1].replace(".p", ""))

            num_episodes = int(files[-1].split(os.sep)[-1].split("_")[0])
            for file in files:
                episode = int(file.split(os.sep)[-1].split("_")[0])
                episode_to_file_paths[episode].append(file)

            for episode in range(num_episodes):
                assert len(episode_to_file_paths[episode]) == 2

        episode_lengths = [len(episode_to_file_paths[episode])
                           for episode in sorted(episode_to_file_paths.keys())]

        # split in train, test, val
        number_of_episodes = len(episode_lengths)
        all_episodes = list(range(number_of_episodes))
        val_start_index = int(number_of_episodes * split[0])
        test_start_index = val_start_index + int(number_of_episodes * split[1])
        train_episodes = all_episodes[:val_start_index]
        val_episodes = all_episodes[val_start_index:test_start_index]
        test_episodes = all_episodes[test_start_index:]

        if train:
            episodes = train_episodes
        elif val:
            episodes = val_episodes
        else:
            episodes = test_episodes

        self.episode_to_file_paths = {}
        self.episode_lengths = []
        for i, episode in enumerate(episodes):
            self.episode_to_file_paths[i] = episode_to_file_paths[episode]
            self.episode_lengths.append(episode_lengths[episode])

    def get_number_of_episodes(self):
        return len(self.episode_lengths)

    def __len__(self):
        return sum(self.episode_lengths)

    def __getitem__(self, idx):
        episode_lengths_cum = np.cumsum(self.episode_lengths)
        episode = np.argwhere(episode_lengths_cum > idx).flatten()[0]
        if episode == 0:
            episode_step = idx
        else:
            episode_step = idx - episode_lengths_cum[episode - 1]

        data = pickle.load(open(self.episode_to_file_paths[episode][episode_step], "rb"))

        background_id = data["info"]["scene_info"]["fixed_objects"]["table"]["uid"]
        camera_idx = 0
        image = np.transpose(data["state_obs"]["rgb_obs"][camera_idx], axes=[2, 0, 1])  # (3, height, width)
        depth = np.expand_dims(data["state_obs"]["depth_obs"][camera_idx], 0)  # (1, height, width)
        normals = np.transpose(data["state_obs"]["normals_obs"][camera_idx], axes=[2, 0, 1])  # (3, height, width)
        occlusion_boundary = np.expand_dims(data["state_obs"]["occlusion_boundary_obs"][camera_idx], 0)  # (1, height, width)
        segmentation_mask = np.expand_dims(data["state_obs"]["segmentation_mask_obs"][camera_idx], 0)  # (1, height, width)
        
        camera_info = data["info"]["camera_info"][camera_idx]

        inputs = (image, depth, segmentation_mask, background_id)
        if self.transform:
            inputs = self.transform(inputs)
        labels = depth, normals, occlusion_boundary, segmentation_mask, background_id
        if self.target_transform:
            labels = self.target_transform(labels)
        return inputs, labels


class InteractionDatasetMasksForCleargrasp(InteractionDataset):

    def __init__(
            self,
            data_folder="dataset",
            transform=None,
            target_transform=None,
            train: bool = False,
            val: bool = False,
            test: bool = False,
            split: tuple = (0.7, 0.2, 0.1),  # (train, val, test)
            transform_cleargrasp = None,
            input_only_cleargrasp = None,
    ):
        super().__init__(data_folder, transform, target_transform,
            train, val, test, split)

        self.transform_cleargrasp = transform_cleargrasp
        self.input_only_cleargrasp = input_only_cleargrasp

    def __getitem__(self, idx):
        ((image, depth), (depth_label, normals, occlusion_boundary, binary_mask)) = super().__getitem__(idx)
        image = np.transpose(image, axes=[1, 2, 0]) # (height, width, 3)
        #normals = np.transpose(normals, axes=[1, 2, 0]) # (height, width, 3)
        
        """print("img dtype", image.dtype) # uint8 (height, width, 3)
        print("img shape", image.shape)
        print("norm dtype", normals.dtype) # float32 (3, height, width)
        print("norm shape", normals.shape)
        print("img min", np.min(image)) # 0
        print("img max", np.max(image)) # 255
        print("norm min", np.min(normals)) # -1
        print("norm max", np.max(normals)) # 1"""

        _img = image
        _label = normals
        # Apply image augmentations and convert to Tensor
        if self.transform_cleargrasp:
            det_tf = self.transform_cleargrasp.to_deterministic()

            _img = det_tf.augment_image(_img)
            
            # Making all values of invalid pixels marked as -1.0 to 0.
            # In raw data, invalid pixels are marked as (-1, -1, -1) so that on conversion to RGB they appear black.
            mask = np.all(_label == -1.0, axis=0)
            _label[:, mask] = 0.0

            _label = _label.transpose((1, 2, 0))  # To Shape: (H, W, 3)
            _label = det_tf.augment_image(_label, hooks=ia.HooksImages(activator=self._activator_masks))
            _label = _label.transpose((2, 0, 1))  # To Shape: (3, H, W)

        # Return Tensors
        _img_tensor = transforms.ToTensor()(_img)

        _label_tensor = torch.from_numpy(_label)
        _label_tensor = nn.functional.normalize(_label_tensor, p=2, dim=0)

        _mask_tensor = torch.ones((1, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)
        
        """print("_img_tensor", _img_tensor.shape) # [3, 256, 256]
        print("_label_tensor", _label_tensor.shape) # [3, 256, 256]
        print("_mask_tensor", _mask_tensor.shape) # [1, 256, 256]
        print("_img_tensor", _img_tensor.dtype) # float32
        print("_label_tensor", _label_tensor.dtype) # float32
        print("_mask_tensor", _mask_tensor.dtype) # float32"""
        
        return _img_tensor, _label_tensor, _mask_tensor

    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only_cleargrasp and augmenter.name in self.input_only_cleargrasp:
            return False
        else:
            return default