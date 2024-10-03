#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import logging
import math
import os
import pathlib
import pickle
from typing import List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from transformers import AutoModel

from dpr.data.biencoder_data import BiEncoderPassage
from dpr.models import init_biencoder_components
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu, setup_logger

from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
)
logger = logging.getLogger()
setup_logger(logger)


def gen_ctx_vectors(
    cfg: DictConfig,
    ctx_rows: List[Tuple[object, BiEncoderPassage]],
    model: nn.Module,
    tensorizer: Tensorizer,
    insert_title: bool = True,
) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = cfg.batch_size
    total = 0
    results = []
    tensor_list = []
    ctx_ids_list = []
    for ctx in ctx_rows:
        tokens_list = tensorizer.text_to_tensor(ctx[1].text, title=ctx[1].title if insert_title else None, ctx_vectors_generate=True)
        for i, tokens in enumerate(tokens_list):
            tensor_list.append(tokens)
            ctx_ids_list.append(ctx[0] + "_" + str(i))
        # for tokens in tokens_list:
        
    for j, batch_start in enumerate(range(0, len(tensor_list), bsz)):
        batch_token_tensors = tensor_list[batch_start : batch_start + bsz]
        ctx_ids = ctx_ids_list[batch_start : batch_start + bsz]
        ctx_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0), cfg.device)
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), cfg.device)
        ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch), cfg.device)
        with torch.no_grad():
            _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
        out = out.cpu()

        # ctx_ids = [r[0] for r in batch]
        extra_info = []

        assert len(ctx_ids) == out.size(0)
        total += len(ctx_ids)

        # TODO: refactor to avoid 'if'
        if extra_info:
            results.extend([(ctx_ids[i], out[i].view(-1).numpy(), *extra_info[i]) for i in range(out.size(0))])
        else:
            results.extend([(ctx_ids[i], out[i].view(-1).numpy()) for i in range(out.size(0))])

        if total % 10 == 0:
            logger.info("Encoded passages chunk %d", total)
    logger.info("Encoded total passages chunk %d", total)
    logger.info("Encoded passages %d", len(ctx_rows))
    return results


@hydra.main(config_path="conf", config_name="gen_embs")
def main(cfg: DictConfig):
    # assert cfg.model_file, "Please specify encoder checkpoint as model_file param"
    assert cfg.ctx_src, "Please specify passages source as ctx_src param"

    cfg = setup_cfg_gpu(cfg)

    #-------------- load dpr checkpoint trained in this repo
    if cfg.is_DPR_checkpoint:
        logger.info("This is a DPR checkpoint")
        saved_state = load_states_from_checkpoint(cfg.model_file)
        set_cfg_params_from_state(saved_state.encoder_params, cfg)
    else: 
        logger.info("This is not DPR checkpoint")
    #--------------
    logger.info("CFG:")
    logger.info("%s", OmegaConf.to_yaml(cfg))
    cfg.encoder.pretrained_model_cfg = cfg.encoder.pretrained_model_cfg.replace("/public/user", "/home")
    tensorizer, encoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True)
    encoder = encoder.ctx_model if cfg.encoder_type == "ctx" else encoder.question_model

    encoder, _ = setup_for_distributed_mode(
        encoder,
        None,
        cfg.device,
        cfg.n_gpu,
        cfg.local_rank,
        cfg.fp16,
        cfg.fp16_opt_level,
    )
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    # --------------------- load dpr checkpoint trained in this repo
    if cfg.is_DPR_checkpoint:
        logger.info("Loading saved model state ...")
        logger.debug("saved model keys =%s", saved_state.model_dict.keys())

        prefix_len = len("ctx_model.")
        ctx_state = {
            key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if key.startswith("ctx_model.")
        }
        model_to_load.load_state_dict(ctx_state, strict=False)
    # --------------------- load other checkpoint or hugginface models
    if not cfg.is_DPR_checkpoint:
        if cfg.from_pretrained:
            model = AutoModel.from_pretrained(cfg.model_file, add_pooling_layer=False)
            model_to_load.load_state_dict(model.state_dict(), strict=True)
        else:
            # T2ranking-----------------
            state_dict = torch.load(cfg.model_file, map_location=lambda storage, loc: storage)
            new_state_dict = {}
            for key in state_dict.keys():
                if key.startswith("module.lm_p.") and "position_ids" not in key:
                    new_state_dict[key[len("module.lm_p."):]] = state_dict[key]
            model_to_load.load_state_dict(new_state_dict, strict=True)
            # --------------------------
            # CaseEncoder---------------
            # parameters = torch.load(cfg.model_file, map_location=lambda storage, loc: storage)
            # case_encoder_key = list(parameters["model"].keys())
            # new_state_dict = {}
            # for key in case_encoder_key:
            #     if key.startswith("model.bert.") and "position_ids" not in key:
            #         new_state_dict[key[len("model.bert."):]] = parameters["model"][key]
            # model_to_load.load_state_dict(new_state_dict, strict=True)
            # --------------------------
    # ---------------------
    logger.info("reading data source: %s", cfg.ctx_src)
    ctx_src = hydra.utils.instantiate(cfg.ctx_sources[cfg.ctx_src])
    all_passages_dict = {}
    ctx_src.load_data_to(all_passages_dict)
    all_passages = [(k, v) for k, v in all_passages_dict.items()]

    shard_size = math.ceil(len(all_passages) / cfg.num_shards)
    start_idx = cfg.shard_id * shard_size
    end_idx = start_idx + shard_size

    logger.info(
        "Producing encodings for passages range: %d to %d (out of total %d)",
        start_idx,
        end_idx,
        len(all_passages),
    )
    shard_passages = all_passages[start_idx:end_idx]

    data = gen_ctx_vectors(cfg, shard_passages, encoder, tensorizer, True)
    file = cfg.out_file + "_" + str(cfg.shard_id)
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info("Writing results to %s" % file)
    with open(file, mode="wb") as f:
        pickle.dump(data, f)

    logger.info("Total passages processed %d. Written to %s", len(data), file)


if __name__ == "__main__":
    main()
