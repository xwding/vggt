# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.layers import Mlp
from vggt.layers.block import Block
from vggt.heads.head_act import activate_pose


class CameraHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    先拿相机 token 做 trunk 编码，再结合上一轮预测结果，对 token 做调制，最后预测一个 delta，不断迭代修正。
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            # absolute translation，绝对平移，3 维
            # quaR：quaternion rotation，四元数旋转，4 维
            # FoV：field of view，视场角，通常 2 维
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        # 平移用什么激活
        # 四元数用什么激活
        # 焦距/视场角用什么激活
        # 比如：
        # 平移可用 linear
        # 四元数可用归一化或线性后再归一化
        # FOV 用 relu 保证为正
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_in,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)  # 对输入的 camera token 先做归一化
        self.trunk_norm = nn.LayerNorm(dim_in)  # 对 trunk 输出再做归一化后再回归 pose

        # Learnable empty camera pose token.
        # 因为迭代 refinement 的第一轮，没有上一轮预测结果可用，所以模型需要一个“起始姿态输入”。
        # 形状是 [1, 1, 9]
        # 第一次迭代时，把它扩展到 [B, S, 9]
        # 再通过 embed_pose 映射到 token 特征维度 dim_in
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        # 这个模块的输入是"上一轮预测的 pose embedding"，输出被分成三部分：
        # 平移项、旋转项和门控项
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        # 它预测的不是最终值，而更像是一个： 当前迭代的 pose 更新量（delta）
        # pk =pk-1 + deltap 卡尔曼滤波器的思想
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=self.target_dim,
            drop=0,
        )

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """
        # Use tokens from the last block for camera prediction.
        # 使用注意力聚合后的最后一层输出的 token 来预测相机参数。
        tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        # 前面 Aggregator 里 token 排列是：[camera_token | register_tokens | patch_tokens]
        # pose_tokens [B, S, C]
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)

        # 最终返回的是一个列表：
        # 第 1 次迭代的 pose [B, S, 9]
        # 第 2 次迭代的 pose [B, S, 9]
        # …
        # 第 num_iterations 次迭代的 pose [B, S, 9]
        pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations)
        return pred_pose_enc_list

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, S, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                #  module_input  [B, S, dim_in]
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                # 这里会把上一轮的 pose 编码拿来作为当前轮的条件输入。
                # detach 避免反向传播跨越所有迭代步骤形成很长的计算图。
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            # poseLN_modulation(module_input) 输出形状是：[B, S, 3 * dim_in] 然后沿最后一维切成三块：
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            # https://zhuanlan.zhihu.com/p/698014972  自适应层归一化（AdaLN）的核心思想是：根据当前迭代的 pose 预测结果，动态调整 trunk 中的特征表示。
            # 1 对 pose_tokens 先做无仿射 LayerNorm
            # 2 用当前轮的 shift/scale 做调制
            # 3 再乘一个 gate
            # 4 最后加回原始 pose_tokens 残差
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens
            # 这一步会让相机 token 之间进行多层 Transformer 交互。
            # 在一段序列的多帧图片的位姿之间做建模。
            pose_tokens_modulated = self.trunk(pose_tokens_modulated)
            # Compute the delta update for the pose encoding.
            # [B, S, 9]
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                # 这不是最终 pose，而是本轮的增量更新：
                # p_t = p_{t-1} + Δp_t
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            # 这一步是把“原始回归值”变成满足物理意义的相机参数。
            # 比如可能会做：
            # 平移：线性或其他范围映射
            # 四元数：归一化
            # FOV：relu 保证为正
            # 因为如果不做这些约束，网络可能输出：
            # 非单位四元数
            # 负的 FOV
            # 不合理的参数范围
            # 所以这里很重要。
            activated_pose = activate_pose(
                pred_pose_enc,
                trans_act=self.trans_act,
                quat_act=self.quat_act,
                fl_act=self.fl_act,
            )
            pred_pose_enc_list.append(activated_pose)

        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift
