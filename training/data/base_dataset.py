# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from PIL import Image, ImageFile

from torch.utils.data import Dataset
from .dataset_util import *

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    """
    Base dataset class for VGGT and VGGSfM training.

    This abstract class handles common operations like image resizing,
    augmentation, and coordinate transformations. Concrete dataset
    implementations should inherit from this class.

    Attributes:
        img_size: Target image size (typically the width)
        patch_size: Size of patches for vit
        augs.scales: Scale range for data augmentation [min, max]
        rescale: Whether to rescale images
        rescale_aug: Whether to apply augmentation during rescaling
        landscape_check: Whether to handle landscape vs portrait orientation
    """

    def __init__(
        self,
        common_conf,
    ):
        """
        Initialize the base dataset with common configuration.

        Args:
            common_conf: Configuration object with the following properties, shared by all datasets:
                - img_size: Default is 518
                - patch_size: Default is 14
                - augs.scales: Default is [0.8, 1.2]
                - rescale: Default is True
                - rescale_aug: Default is True
                - landscape_check: Default is True
        """
        super().__init__()
        self.img_size = common_conf.img_size
        self.patch_size = common_conf.patch_size
        self.aug_scale = common_conf.augs.scales
        self.rescale = common_conf.rescale
        self.rescale_aug = common_conf.rescale_aug
        self.landscape_check = common_conf.landscape_check

    def __len__(self):
        return self.len_train

    def __getitem__(self, idx_N):
        """
        Get an item from the dataset.

        Args:
            idx_N: Tuple containing (seq_index, img_per_seq, aspect_ratio)

        Returns:
            Dataset item as returned by get_data()
        """
        seq_index, img_per_seq, aspect_ratio = idx_N
        return self.get_data(seq_index=seq_index, img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)

    def get_data(self, seq_index=None, seq_name=None, ids=None, aspect_ratio=1.0):
        """
        Abstract method to retrieve data for a given sequence.

        Args:
            seq_index (int, optional): Index of the sequence
            seq_name (str, optional): Name of the sequence
            ids (list, optional): List of frame IDs
            aspect_ratio (float, optional): Target aspect ratio.

        Returns:
            Dataset-specific data

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError(
            "This is an abstract method and should be implemented in the subclass, i.e., each dataset should implement its own get_data method."
        )

    def get_target_shape(self, aspect_ratio):
        """
        Calculate the target shape based on the given aspect ratio.

        Args:
            aspect_ratio: Target aspect ratio

        Returns:
            numpy.ndarray: Target image shape [height, width]
        """
        short_size = int(self.img_size * aspect_ratio)
        small_size = self.patch_size

        # ensure the input shape is friendly to vision transformer
        if short_size % small_size != 0:
            short_size = (short_size // small_size) * small_size

        image_shape = np.array([short_size, self.img_size])
        return image_shape

    def process_one_image(
        self,
        image,  # RGB 图像
        depth_map,  # 深度图
        extri_opencv,  # 外参，描述相机位姿
        intri_opencv,  # 内参，描述相机成像模型
        original_size,  # 原图尺寸
        target_image_shape,  # 模型希望的目标尺寸
        track=None,  # 可选的跟踪点
        filepath=None,
        safe_bound=4,
    ):
        """
        Process a single image and its associated data.

        This method handles image transformations, depth processing, and coordinate conversions.

        Args:
            image (numpy.ndarray): Input image array
            depth_map (numpy.ndarray): Depth map array
            extri_opencv (numpy.ndarray): Extrinsic camera matrix (OpenCV convention)
            intri_opencv (numpy.ndarray): Intrinsic camera matrix (OpenCV convention)
            original_size (numpy.ndarray): Original image size [height, width]
            target_image_shape (numpy.ndarray): Target image shape after processing
            track (numpy.ndarray, optional): Optional tracking information. Defaults to None.
            filepath (str, optional): Optional file path for debugging. Defaults to None.
            safe_bound (int, optional): Safety margin for cropping operations. Defaults to 4.

        Returns:
            tuple: (
                image (numpy.ndarray): Processed image,
                depth_map (numpy.ndarray): Processed depth map,
                extri_opencv (numpy.ndarray): Updated extrinsic matrix,
                intri_opencv (numpy.ndarray): Updated intrinsic matrix,
                world_coords_points (numpy.ndarray): 3D points in world coordinates,
                cam_coords_points (numpy.ndarray): 3D points in camera coordinates,
                point_mask (numpy.ndarray): Boolean mask of valid points,
                track (numpy.ndarray, optional): Updated tracking information
            )
        """
        # 拷贝输入，避免后续裁剪、缩放、旋转等操作原地修改数据集缓存或调用方持有的数组。
        # image/depth_map 是像素网格数据；extri/intri 是相机参数，后续几何变换会同步更新 intri，
        # 某些旋转操作也会更新 extri，所以这里都先 copy 一份。
        image = np.copy(image)
        depth_map = np.copy(depth_map)
        extri_opencv = np.copy(extri_opencv)
        intri_opencv = np.copy(intri_opencv)
        if track is not None:
            # track 通常是二维像素轨迹点，图像裁剪、缩放或旋转时也要同步变换。
            track = np.copy(track)

        # 训练阶段的数据增强：随机生成一个临时裁剪尺寸 aug_size。
        # self.aug_scale 是 [min_scale, max_scale]，分别对高度和宽度采样。
        # 注意这里把随机比例限制到最大 1.0，意味着只会裁小或保持原尺寸，避免需要向外补 padding。
        if self.training and self.aug_scale:
            random_h_scale, random_w_scale = np.random.uniform(self.aug_scale[0], self.aug_scale[1], 2)
            # 如果采样到大于 1 的比例，直接截断为 1，保证 aug_size 不超过原图尺寸。
            random_h_scale = min(random_h_scale, 1.0)
            random_w_scale = min(random_w_scale, 1.0)
            aug_size = original_size * np.array([random_h_scale, random_w_scale])
            aug_size = aug_size.astype(np.int32)
        else:
            # 验证/测试阶段，或者没有配置 scale augmentation 时，不做随机尺度裁剪。
            aug_size = original_size

        # 第一次基于主点的裁剪：
        # crop_image_depth_and_intrinsic_by_pp 会围绕相机主点 principal point 做裁剪，
        # 尽量把主点移动到新图像中心附近。这样能减少主点偏移过大的样本对训练的影响。
        #
        # 这个函数会同步处理：
        # - image: 裁剪 RGB 图像
        # - depth_map: 用相同窗口裁剪深度图，保证深度仍与 RGB 对齐
        # - intri_opencv: 根据裁剪偏移更新 cx/cy
        # - track: 如果存在，也把轨迹点坐标减去裁剪偏移
        image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            image,
            depth_map,
            intri_opencv,
            aug_size,
            track=track,
            filepath=filepath,
        )

        # 第一次裁剪之后，图像尺寸已经变化，后续 resize/crop 都应以当前尺寸为准。
        original_size = np.array(image.shape[:2])  # update original_size
        target_shape = target_image_shape

        # 可选的横竖屏处理：
        # 如果开启 landscape_check，并且当前图像明显是竖长图，高度超过宽度 1.25 倍，
        # 同时目标 shape 不是正方形，则有 50% 概率交换目标高宽。
        # 后面 resize 会先按交换后的目标尺寸处理，最后再通过 90 度旋转回模型期望方向。
        rotate_to_portrait = False
        if self.landscape_check:
            if original_size[0] > 1.25 * original_size[1]:
                if (target_image_shape[0] != target_image_shape[1]) and (np.random.rand() > 0.5):
                    target_shape = np.array([target_image_shape[1], target_image_shape[0]])
                    rotate_to_portrait = True

        # resize 到目标尺寸，并同步更新相机内参。
        # 对图像/深度图来说，这是普通的空间缩放；对相机内参来说，需要按缩放比例更新：
        # - fx/fy 随宽高缩放比例变化
        # - cx/cy 也随坐标系缩放变化
        #
        # rescale_aug=True 时，resize_image_depth_and_intrinsic 内部可能带有额外随机缩放/裁剪策略；
        # safe_bound 用于避免采样或裁剪时过于贴边。
        if self.rescale:
            image, depth_map, intri_opencv, track = resize_image_depth_and_intrinsic(
                image,
                depth_map,
                intri_opencv,
                target_shape,
                original_size,
                track=track,
                safe_bound=safe_bound,
                rescale_aug=self.rescale_aug,
            )
        else:
            print("Not rescaling the images")

        # 最终严格裁剪到 target_shape。
        # resize 后可能因为保持比例、取整或 patch 对齐等原因，尺寸仍略大于目标尺寸。
        # strict=True 表示这里必须得到精确目标大小，否则后续 batch stack 会因为尺寸不一致失败。
        # 内参的主点坐标会再次根据最终裁剪窗口更新。
        image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            image,
            depth_map,
            intri_opencv,
            target_shape,
            track=track,
            filepath=filepath,
            strict=True,
        )

        # 如果前面为了横竖屏增强交换过目标高宽，这里做一次 90 度旋转。
        # rotate_90_degrees 会同步更新：
        # - image/depth_map 的像素排列
        # - intri_opencv 的 fx/fy/cx/cy 和图像坐标定义
        # - extri_opencv 的相机坐标轴方向，使外参仍符合旋转后的 OpenCV 相机约定
        # - track 的二维坐标
        if rotate_to_portrait:
            assert self.landscape_check
            clockwise = np.random.rand() > 0.5
            image, depth_map, extri_opencv, intri_opencv, track = rotate_90_degrees(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                clockwise=clockwise,
                track=track,
            )

        # 把深度图反投影成每个像素对应的 3D 点。
        #
        # 对每个有效深度像素 (u, v, z)，先用内参 K 还原相机坐标：
        #   x = (u - cx) / fx * z
        #   y = (v - cy) / fy * z
        #   z = depth
        # 得到 cam_coords_points。
        #
        # 然后使用 OpenCV convention 的外参 extri_opencv（camera-from-world）把相机坐标变换回世界坐标，
        # 得到 world_coords_points。point_mask 标记哪些像素深度有效、反投影结果有效。
        world_coords_points, cam_coords_points, point_mask = depth_to_world_coords_points(
            depth_map, extri_opencv, intri_opencv
        )

        return (
            image,  # 处理后的图像和深度图
            depth_map,
            extri_opencv,  # 对应更新后的内参/外参
            intri_opencv,  # 对应更新后的内参/外参
            world_coords_points,  # 每个像素对应的 3D 点（世界坐标系）
            cam_coords_points,  # 每个像素对应的 3D 点（相机坐标系）
            point_mask,  # 每个像素的有效性掩码
            track,  # 可选的跟踪点
        )

    def get_nearby_ids(self, ids, full_seq_num, expand_ratio=None, expand_range=None):
        """
        TODO: add the function to sample the ids by pose similarity ranking.

        Sample a set of IDs from a sequence close to a given start index.

        You can specify the range either as a ratio of the number of input IDs
        or as a fixed integer window.


        Args:
            ids (list): Initial list of IDs. The first element is used as the anchor.
            full_seq_num (int): Total number of items in the full sequence.
            expand_ratio (float, optional): Factor by which the number of IDs expands
                around the start index. Default is 2.0 if neither expand_ratio nor
                expand_range is provided.
            expand_range (int, optional): Fixed number of items to expand around the
                start index. If provided, expand_ratio is ignored.

        Returns:
            numpy.ndarray: Array of sampled IDs, with the first element being the
                original start index.

        Examples:
            # Using expand_ratio (default behavior)
            # If ids=[100,101,102] and full_seq_num=200, with expand_ratio=2.0,
            # expand_range = int(3 * 2.0) = 6, so IDs sampled from [94...106] (if boundaries allow).

            # Using expand_range directly
            # If ids=[100,101,102] and full_seq_num=200, with expand_range=10,
            # IDs are sampled from [90...110] (if boundaries allow).

        Raises:
            ValueError: If no IDs are provided.
        """
        if len(ids) == 0:
            raise ValueError("No IDs provided.")

        if expand_range is None and expand_ratio is None:
            expand_ratio = 2.0  # Default behavior

        total_ids = len(ids)
        start_idx = ids[0]

        # Determine the actual expand_range
        if expand_range is None:
            # Use ratio to determine range
            expand_range = int(total_ids * expand_ratio)

        # Calculate valid boundaries
        low_bound = max(0, start_idx - expand_range)
        high_bound = min(full_seq_num, start_idx + expand_range)

        # Create the valid range of indices
        valid_range = np.arange(low_bound, high_bound)

        # Sample 'total_ids - 1' items, because we already have the start_idx
        sampled_ids = np.random.choice(
            valid_range,
            size=(total_ids - 1),
            replace=True,  # we accept the situation that some sampled ids are the same
        )

        # Insert the start_idx at the beginning
        result_ids = np.insert(sampled_ids, 0, start_idx)

        return result_ids
