import torch
import numpy as np
import noise
import random
import cv2 as cv
import math
import matplotlib.pyplot as plt


class ToFloatImageFromUint8Image:
    def __call__(self, img):
        img = torch.as_tensor(img, dtype=torch.float32) / 255.0
        return img

class BinaryObjectMask:
    """
    Transforms the segmentation mask to an binary object mask, 0 = background, 1 = any object
    """
    def __call__(self, labels):
        depth, normals, occlusion_boundary, segmentation_mask, background_id = labels
        assert background_id != 0 # avoid to first copy the segmentation_mask
        segmentation_mask[segmentation_mask != background_id] = 1
        segmentation_mask[segmentation_mask == background_id] = 0
        return depth, normals, occlusion_boundary, segmentation_mask


class DepthCameraNoise:
    """
    Applies synthetic noise on the depth image.
    """

    @staticmethod
    def generate_perlin_noise(shape):
        scale = 50.0
        octaves = 4
        persistence = 0.5
        lacunarity = 1.9
        seed = np.random.randint(1000)

        noise_img = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                noise_img[i][j] = noise.pnoise2(i / scale,
                                                j / scale,
                                                octaves=octaves,
                                                persistence=persistence,
                                                lacunarity=lacunarity,
                                                repeatx=shape[0],
                                                repeaty=shape[1],
                                                base=seed)
        noise_img -= np.min(noise_img)
        noise_img /= np.max(noise_img)

        noise_mask = np.zeros_like(noise_img, dtype=bool)
        noise_mask[noise_img > 0.6] = True
        return noise_mask

    @staticmethod
    def convolve_random_filter(x, kernel_dim):
        """
        2d convolution

        Args:
            x: image (C, H, W)
            kernel: (H, W)
        """
        feature_map = np.empty_like(x)
        kernel_height, kernel_width = kernel_dim
        pad_h = (kernel_height - 1) // 2
        pad_w = (kernel_width - 1) // 2
        x = cv.copyMakeBorder(x.transpose((1, 2, 0)), pad_h, pad_h, pad_w, pad_w, cv.BORDER_REFLECT_101)
        x = np.expand_dims(x, 0)
        height = x.shape[1]
        width = x.shape[2]
        # iterate over input height and width with stride
        for i in range(0, height - kernel_height + 1):
            for j in range(0, width - kernel_width + 1):
                chunk = x[..., i:i + kernel_height, j:j + kernel_width]
                kernel = np.random.random((kernel_height, kernel_width))
                kernel = (kernel > 0.5) / (kernel_height * kernel_width)
                feature_map[..., i, j] = np.sum(chunk * np.expand_dims(kernel, axis=0), axis=(0, 1, 2))
        return feature_map

    def __call__(self, inputs):
        image, depth, segmentation_mask, background_id = inputs
        depth_noisy = depth.copy()
        instance_segmentation_frame = np.transpose(segmentation_mask, (1, 2, 0))  # (C, H, W) -> (H, W, C)

        object_mask = np.all(instance_segmentation_frame != background_id, axis=2)

        # extract edges
        depth_noisy = depth_noisy[0, ...]  # (1, H, W) -> (H, W)
        edges = cv.Laplacian(depth_noisy, -1, ksize=5, scale=1)
        edges_threshold = (edges < -0.5) * (1 - edges)
        edges_thick = cv.dilate(edges_threshold, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
        edges_thick = cv.blur(edges_thick, (3, 3))  # blur to get higher values in middle of edge

        # add noise with a random kernel
        edges_mask = edges_thick > 1.0
        edges_noise = self.convolve_random_filter(np.expand_dims(edges_mask, 0).astype(float), (5, 5))
        edges_noise = edges_noise[0, ...]
        edges_mask_noise = edges_noise > 0.1

        # remove holes from noise
        edges = cv.dilate(edges_mask_noise.astype(float), cv.getStructuringElement(cv.MORPH_RECT, (2, 2)))
        edges = cv.erode(edges.astype(float), cv.getStructuringElement(cv.MORPH_RECT, (5, 5)))

        # apply shadow on depth image
        depth_noisy[edges > 0] = 0.0
        depth_noisy = np.expand_dims(depth_noisy, 0)

        # apply perlin noise on depth image with object mask
        noise_mask = self.generate_perlin_noise(depth.shape[1:])
        depth_noisy[:, np.bitwise_and(noise_mask, object_mask)] = 0.0

        return image, depth_noisy # inputs, labels
