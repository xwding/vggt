# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os.path as osp
import os
import logging
from typing import Iterable, List, Tuple

import cv2
import random
import numpy as np


from data.dataset_util import *
from data.dataset_util import read_depth, read_image_cv2, threshold_depth_map
from data.base_dataset import BaseDataset
from typing import Optional

SEEN_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]


class Co3dDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        CO3D_DIR: Optional[str] = None,
        CO3D_ANNOTATION_DIR: Optional[str] = None,
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        verify_frame_files: bool = True,
        available_categories_only: bool = True,
    ):
        """
        Initialize the Co3dDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            CO3D_DIR (str): Directory path to CO3D data.
            CO3D_ANNOTATION_DIR (str): Directory path to CO3D annotations.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If CO3D_DIR or CO3D_ANNOTATION_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.verify_frame_files = verify_frame_files
        self.available_categories_only = available_categories_only

        if CO3D_DIR is None or CO3D_ANNOTATION_DIR is None:
            raise ValueError("Both CO3D_DIR and CO3D_ANNOTATION_DIR must be specified.")

        category = sorted(SEEN_CATEGORIES)

        if self.debug:
            category = ["apple"]

        if split == "train":
            split_name_list = ["train"]
            self.len_train = len_train
        elif split == "test":
            split_name_list = ["test"]
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.invalid_sequence: List[str] = []  # set any invalid sequence names here

        self.category_map: dict = {}
        self.data_store: dict = {}
        self.seqlen = None
        self.min_num_images = min_num_images

        logging.info(f"CO3D_DIR is {CO3D_DIR}")

        self.CO3D_DIR = CO3D_DIR
        self.CO3D_ANNOTATION_DIR = CO3D_ANNOTATION_DIR

        if self.available_categories_only:
            available_categories = self._get_available_categories(category)
            skipped_categories = sorted(set(category) - set(available_categories))
            if skipped_categories:
                logging.warning(
                    "Skipping CO3D categories without local data: %s",
                    ", ".join(skipped_categories),
                )
            category = available_categories

        if not category:
            raise ValueError(
                f"No CO3D categories are available under {self.CO3D_DIR}. "
                "Please verify the dataset path and extracted category folders."
            )

        total_frame_num = 0
        total_missing_frames = 0
        total_short_sequences = 0

        for c in category:
            for split_name in split_name_list:
                annotation_file = osp.join(self.CO3D_ANNOTATION_DIR, f"{c}_{split_name}.jgz")

                try:
                    with gzip.open(annotation_file, "r") as fin:
                        annotation = json.loads(fin.read())
                except FileNotFoundError:
                    logging.error(f"Annotation file not found: {annotation_file}")
                    continue

                for seq_name, seq_data in annotation.items():
                    if seq_name in self.invalid_sequence:
                        continue

                    valid_seq_data, missing_frames = self._filter_valid_frames(seq_data)
                    total_missing_frames += missing_frames

                    if len(valid_seq_data) < min_num_images:
                        total_short_sequences += 1
                        continue

                    total_frame_num += len(valid_seq_data)
                    self.data_store[seq_name] = valid_seq_data

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        if self.sequence_list_len == 0:
            raise ValueError(
                "No valid CO3D sequences were found after filtering local files. "
                f"Checked data root: {self.CO3D_DIR}, annotation root: {self.CO3D_ANNOTATION_DIR}, "
                f"required minimum frames per sequence: {self.min_num_images}."
            )

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Co3D Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Co3D valid frames kept: {self.total_frame_num}")
        if total_missing_frames > 0:
            logging.warning(
                "%s: Skipped %d frames whose local image/depth/mask files are missing.",
                status,
                total_missing_frames,
            )
        if total_short_sequences > 0:
            logging.warning(
                "%s: Skipped %d sequences with fewer than %d valid local frames.",
                status,
                total_short_sequences,
                self.min_num_images,
            )
        logging.info(f"{status}: Co3D Data dataset length: {len(self)}")

    def _get_available_categories(self, categories: Iterable[str]) -> List[str]:
        return [c for c in categories if osp.isdir(osp.join(self.CO3D_DIR, c))]

    def _build_frame_paths(self, filepath: str) -> Tuple[str, str, str]:
        image_path = osp.join(self.CO3D_DIR, filepath)
        depth_path = image_path.replace("/images", "/depths") + ".geometric.png"
        depth_mask_path = image_path.replace("/images", "/depth_masks").replace(".jpg", ".png")
        return image_path, depth_path, depth_mask_path

    def _frame_assets_exist(self, filepath: str) -> bool:
        image_path, depth_path, depth_mask_path = self._build_frame_paths(filepath)

        if not osp.exists(image_path):
            return False

        if self.load_depth and (not osp.exists(depth_path) or not osp.exists(depth_mask_path)):
            return False

        return True

    def _filter_valid_frames(self, seq_data: list) -> Tuple[list, int]:
        if not self.verify_frame_files:
            return seq_data, 0

        valid_seq_data = []
        missing_frames = 0

        for anno in seq_data:
            filepath = anno.get("filepath")
            if filepath is None or not self._frame_assets_exist(filepath):
                missing_frames += 1
                continue
            valid_seq_data.append(anno)

        return valid_seq_data, missing_frames

    def get_data(
        self,
        seq_index: Optional[int] = None,
        seq_name: Optional[str] = None,
        ids: Optional[np.ndarray | List[int]] = None,
        aspect_ratio: float = 1.0,
        img_per_seq: Optional[int] = None,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.sequence_list_len == 0:
            raise RuntimeError("CO3D dataset is empty after filtering unavailable local files.")

        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_index is None and seq_name is None:
            raise ValueError("Either seq_index or seq_name must be provided.")

        if img_per_seq is None:
            raise ValueError("img_per_seq must be provided when sampling a CO3D sequence.")

        if seq_name is None:
            assert seq_index is not None
            seq_name = self.sequence_list[seq_index]

        assert seq_name is not None

        metadata = self.data_store[seq_name]

        if ids is None:
            ids = np.random.choice(len(metadata), img_per_seq, replace=self.allow_duplicate_img)

        assert ids is not None
        annos = [metadata[i] for i in ids]

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []

        for anno in annos:
            filepath = anno["filepath"]

            image_path, depth_path, mvs_mask_path = self._build_frame_paths(filepath)
            image = read_image_cv2(image_path)

            if image is None:
                raise FileNotFoundError(
                    f"Failed to load CO3D image file: {image_path}. "
                    "Please re-check the extracted subset or enable sequence filtering."
                )

            if self.load_depth:
                depth_map = read_depth(depth_path, 1.0)

                mvs_mask = cv2.imread(mvs_mask_path, cv2.IMREAD_GRAYSCALE)
                if depth_map is None or mvs_mask is None:
                    raise FileNotFoundError(
                        f"Failed to load CO3D depth assets for {image_path}. "
                        f"depth={depth_path}, mask={mvs_mask_path}"
                    )

                mvs_mask = mvs_mask > 128
                depth_map[~mvs_mask] = 0

                depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=98)
            else:
                depth_map = None

            original_size = np.array(image.shape[:2])
            extri_opencv = np.array(anno["extri"])
            intri_opencv = np.array(anno["intri"])

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=filepath,
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)

        set_name = "co3d"

        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        return batch
