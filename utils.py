import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
import mmcv
import os.path as osp

import numpy as np


def save_checkpoint_on_pavi(local_filename, pavi_filename):
    assert pavi_filename.startswith('pavi://')
    try:
        from pavi import modelcloud
    except ImportError:
        raise ImportError(
            'Please install pavi to load checkpoint from modelcloud.')

    try:
        from pavi import modelcloud
        from pavi import exception
    except ImportError:
        raise ImportError(
            'Please install pavi to load checkpoint from modelcloud.')
    model_path = pavi_filename[7:]
    model_dir, model_name = osp.split(model_path)
    model = modelcloud.get(model_dir, node_type=4, automake=True)

    try:
        model.create_file(local_filename, version='2.0')
    except exception.NodeDuplicatedError:
        #ckpt_file = modelcloud.get(model_path, node_type=6)
        #ckpt_file.delete()
        #modelcloud.delete(pavi_filename, node_type=4)
        #print(model_path)
        ckpt_file = modelcloud.get(model_path, version='2.0')
        ckpt_file.delete()
        model.create_file(local_filename, version='2.0')


def update_from_config(args):
    cfg = mmcv.Config.fromfile(args.config)
    for _, cfg_item in cfg._cfg_dict.items():
        for k, v in cfg_item.items():
            setattr(args, k, v)
    if args.output_dir == '':
        config_name = args.config.split('/')[-1].replace('.py', '')
        args.output_dir = osp.join('checkpoints', config_name)
    if args.resume == '':
        args.resume = osp.join(args.output_dir, 'checkpoint.pth')

    return args


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if False:
        # if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(proc_id % num_gpus)
        import subprocess
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        # specify master port
        os.environ['MASTER_PORT'] = str(args.port)
        # use MASTER_ADDR in the environment variable if it already exists
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['RANK'] = str(proc_id)
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    if 'SLURM_PROCID' in os.environ:
        torch.distributed.init_process_group(backend=args.dist_backend)
    else:
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
