# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from dinov2.utils import utils

from . import vision_transformer as vits

logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        if args.get("temp", None):
            vit_kwargs.update({"temp": args.get("temp")})
        if args.get("L", None):
            vit_kwargs.update({"L": args.get("L")})
        if args.get("V", None):
            vit_kwargs.update({"V": args.get("V")})
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    img_size = cfg.MODEL.img_size or cfg.crops.global_crops_size
    if only_teacher:
        return build_model(cfg.student, only_teacher=only_teacher, img_size=img_size)
    student, teacher, embed_dim = build_model(
        cfg.student, only_teacher=only_teacher, img_size=img_size
    )
    if cfg["MODEL"]["pretrained_backbone"]:
        path = cfg["MODEL"]["pretrained_backbone"]
        utils.load_pretrained_weights(student, path, "teacher")
        utils.load_pretrained_weights(teacher, path, "teacher")
    return student, teacher, embed_dim
