import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import glob
import itertools
from collections import defaultdict
import pickle
from torchvision.transforms import transforms
from imgaug import augmenters as iaa
import imgaug as ia

import sys
sys.path.append('../../../data')
from transforms import ToFloatImageFromUint8Image
from dataset import InteractionDataset

class InteractionDatasetEvalForCleargrasp(InteractionDataset):

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

        depth = np.transpose(depth, axes=[1, 2, 0]) # (height, width, 1)
        depth = np.squeeze(depth, 2) # (height, width)

        depth_label = np.transpose(depth_label, axes=[1, 2, 0]) # (height, width, 1)
        depth_label = np.squeeze(depth_label, 2) # (height, width)

        binary_mask = np.transpose(binary_mask, axes=[1, 2, 0]) # (height, width, 1)
        binary_mask = np.squeeze(binary_mask, 2) # (height, width)
        binary_mask = binary_mask.astype(bool)

        return ((image, depth), (depth_label, normals, occlusion_boundary, binary_mask))


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
        binary_mask = np.transpose(binary_mask, axes=[1, 2, 0]) # (height, width, 1)
        binary_mask = np.squeeze(binary_mask, 2) # (height, width)
        binary_mask = binary_mask.astype(np.uint8)
        
        """
        print("img dtype", image.dtype) # uint8 (height, width, 3)
        print("img shape", image.shape)
        print("bin dtype", binary_mask.dtype) # uint8 (height, width)
        print("bin shape", binary_mask.shape)
        print("img min", np.min(image)) # 0
        print("img max", np.max(image)) # 255
        print("unique", np.unique(binary_mask)) # [0, 1]
        """

        _img = image
        _label = binary_mask
        # Apply image augmentations and convert to Tensor
        if self.transform_cleargrasp:
            det_tf = self.transform_cleargrasp.to_deterministic()
            _img = det_tf.augment_image(_img)
            _img = np.ascontiguousarray(_img)  # To prevent errors from negative stride, as caused by fliplr()
        
        _label = det_tf.augment_image(_label, hooks=ia.HooksImages(activator=self._activator_masks))

        # Return Tensors
        _img_tensor = transforms.ToTensor()(_img)

        _label_tensor = torch.from_numpy(_label.astype(np.float32))
        _label_tensor = torch.unsqueeze(_label_tensor, 0)

        """
        print("_img_tensor", _img_tensor.shape) # [3, 256, 256]
        print("_label_tensor", _label_tensor.shape) # [1, 256, 256]
        print("_img_tensor", _img_tensor.dtype) # float32
        print("_label_tensor", _label_tensor.dtype) # float32
        """

        return _img_tensor, _label_tensor

    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only_cleargrasp and augmenter.name in self.input_only_cleargrasp:
            return False
        else:
            return default


class InteractionDatasetOcclusionBoundariesForCleargrasp(InteractionDataset):

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
        occlusion_boundary = np.transpose(occlusion_boundary, axes=[1, 2, 0]) # (height, width, 1)
        
        """print("img dtype", image.dtype) # uint8 (height, width, 3)
        print("img shape", image.shape)
        print("occ dtype", occlusion_boundary.dtype) # uint8 (height, width, 1)
        print("occ shape", occlusion_boundary.shape)
        print("img min", np.min(image)) # 0
        print("img max", np.max(image)) # 255
        print("unique", np.unique(occlusion_boundary)) # [0, 1, 2]"""

        _img = image
        _label = occlusion_boundary
        # Apply image augmentations and convert to Tensor
        if self.transform_cleargrasp:
            det_tf = self.transform_cleargrasp.to_deterministic()
            _img = det_tf.augment_image(_img)
            _img = np.ascontiguousarray(_img)  # To prevent errors from negative stride, as caused by fliplr()
        
        _label = det_tf.augment_image(_label, hooks=ia.HooksImages(activator=self._activator_masks))

        # Return Tensors
        _img_tensor = transforms.ToTensor()(_img)

        _label_tensor = transforms.ToTensor()(_label.astype(np.float))

        """print("_img_tensor", _img_tensor.shape) # [3, 256, 256]
        print("_label_tensor", _label_tensor.shape) # [1, 256, 256]
        print("_img_tensor", _img_tensor.dtype) # float32
        print("_label_tensor", _label_tensor.dtype) # float64"""

        return _img_tensor, _label_tensor

    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only_cleargrasp and augmenter.name in self.input_only_cleargrasp:
            return False
        else:
            return default


class InteractionDatasetSurfaceNormalsForCleargrasp(InteractionDataset):

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