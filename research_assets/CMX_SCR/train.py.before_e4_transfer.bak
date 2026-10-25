import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from config import config
from dataloader.dataloader import get_train_loader
from models.builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
logger = get_logger()

os.environ['MASTER_PORT'] = '169710'

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset)

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    
    model=segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)

    # E3: initialize compatible model weights from E2 only.
    # Do not restore E2 optimizer, epoch, or training state.
    if config.backbone == 'mit_b2_e3':
        import torch

        e2_checkpoint_path = r"/home/cse-sdpl/Downloads/CMX_E2/log_TDD_mit_b2_e2/checkpoint/epoch-100.pth"
        e2_checkpoint = torch.load(
            e2_checkpoint_path,
            map_location="cpu",
            weights_only=False
        )
        e2_state = e2_checkpoint["model"]
        e3_state = model.state_dict()

        compatible_state = {
            key: value for key, value in e2_state.items()
            if key in e3_state and e3_state[key].shape == value.shape
        }

        result = model.load_state_dict(compatible_state, strict=False)

        # Only E3 reliability-head parameters should be missing.
        assert not result.unexpected_keys, result.unexpected_keys
        assert all(
            ".reliability_head." in key
            for key in result.missing_keys
        ), result.missing_keys

        # Ensure reliability heads start with equal modality weights.
        for module in model.backbone.FFMs:
            final_conv = module.reliability_head[-1]
            torch.nn.init.zeros_(final_conv.weight)
            torch.nn.init.zeros_(final_conv.bias)

        print(
            f"E3 initialized from E2: {len(compatible_state)} tensors; "
            f"new reliability parameters: {len(result.missing_keys)}"
        )

    
    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr
    
    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)

    # E2 FIX: include standalone learnable fusion parameters in the optimizer.
    # group_weight() handles module weights/biases but can miss raw nn.Parameters.
    optimizer_param_ids = {
        id(param)
        for group in params_list
        for param in group['params']
    }
    missing_params = [
        param for param in model.parameters()
        if param.requires_grad and id(param) not in optimizer_param_ids
    ]

    if missing_params:
        params_list.append({
            'params': missing_params,
            'lr': base_lr,
            'weight_decay': 0.0,
        })
        logger.info(
            f"E2 optimizer fix: added {len(missing_params)} missing trainable "
            "parameter tensors with zero weight decay."
        )

    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError


    # E3 runtime optimizer audit (diagnostic only)
    if config.backbone == 'mit_b2_e3':
        all_trainable = {
            id(p): name for name, p in model.named_parameters()
            if p.requires_grad
        }
        opt_ids = [
            id(p)
            for group in optimizer.param_groups
            for p in group['params']
        ]
        missing = [
            name for pid, name in all_trainable.items()
            if pid not in set(opt_ids)
        ]
        duplicates = len(opt_ids) - len(set(opt_ids))
        reliability = [
            name for name, p in model.named_parameters()
            if p.requires_grad and '.reliability_head.' in name
        ]

        print("E3 trainable tensors:", len(all_trainable))
        print("E3 optimizer tensors:", len(opt_ids))
        print("E3 reliability-head tensors:", len(reliability))
        print("E3 parameters missing from optimizer:", missing)
        print("E3 duplicate optimizer entries:", duplicates)

        assert not missing, f"Missing optimizer parameters: {missing}"
        assert duplicates == 0, "Duplicate optimizer parameters found"
        assert len(reliability) > 0, "Reliability heads not found"
        print("E3 optimizer audit: PASS")

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                            output_device=engine.local_rank, find_unused_parameters=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()
    logger.info('begin trainning:')
    
    for epoch in range(engine.state.epoch, config.nepochs+1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0

        for idx in pbar:
            engine.update_iteration(epoch, idx)

            minibatch = next(dataloader)
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)

            aux_rate = 0.2
            loss = model(imgs, modal_xs, gts)

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch- 1) * config.niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
            else:
                sum_loss += loss
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))

            del loss
            pbar.set_description(print_str, refresh=False)
        
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)

        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
