import argparse
import mmcv
import os
import os.path as osp
import sys
import time
import torch
import torch.distributed as dist
from mmcv import Config
from mmcv import digit_version as dv
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash

# Allow this file to be launched directly without setting PYTHONPATH.
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from protogcn import __version__
from protogcn.apis import init_random_seed, train_model
from protogcn.datasets import build_dataset
from protogcn.models import build_model
from protogcn.utils import collect_env, get_root_logger, mc_off, mc_on, test_port


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test-last',
        action='store_true',
        help='whether to test the checkpoint after training')
    parser.add_argument(
        '--test-best',
        action='store_true',
        help='whether to test the best checkpoint (if applicable) after training')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--ann-file', help='dataset annotation file')
    parser.add_argument('--work-dir', help='directory for logs and checkpoints')
    parser.add_argument(
        '--batch-size-per-gpu', type=int, help='training batch size per GPU')
    parser.add_argument(
        '--workers-per-gpu', type=int, help='data loader workers per GPU')
    parser.add_argument('--learning-rate', type=float, help='optimizer learning rate')
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--log-interval', type=int, help='iterations between logs')
    parser.add_argument('--wandb-project', help='Weights & Biases project name')
    parser.add_argument('--wandb-name', help='Weights & Biases run name')
    parser.add_argument('--wandb-entity', help='Weights & Biases entity name')
    parser.add_argument(
        '--wandb-mode', choices=['online', 'offline'], default='online',
        help='Weights & Biases logging mode')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['pytorch', 'slurm'],
        default='pytorch',
        help='job launcher')
    parser.add_argument(
        '--compile',
        action='store_true',
        help='whether to compile the model before training / testing (only available in pytorch 2.0)')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--local-rank', type=int, default=-1)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.ann_file is not None:
        for split in ('train', 'val', 'test'):
            cfg.data[split].ann_file = args.ann_file
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.batch_size_per_gpu is not None:
        cfg.data.videos_per_gpu = args.batch_size_per_gpu
    if args.workers_per_gpu is not None:
        cfg.data.workers_per_gpu = args.workers_per_gpu
    if args.learning_rate is not None:
        cfg.optimizer.lr = args.learning_rate
    if args.epochs is not None:
        cfg.total_epochs = args.epochs
    if args.log_interval is not None:
        cfg.log_config.interval = args.log_interval
    if args.wandb_project is not None:
        init_kwargs = dict(
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config=dict(
                model='ProtoGCN',
                dataset='NTU120-3D',
                config=args.config,
                batch_size_per_gpu=cfg.data.videos_per_gpu,
                world_size=int(os.environ.get('WORLD_SIZE', '1')),
                learning_rate=cfg.optimizer.lr,
                epochs=cfg.total_epochs,
                seed=args.seed))
        if args.wandb_entity is not None:
            init_kwargs['entity'] = args.wandb_entity
        cfg.log_config.hooks.append(
            dict(type='WandbLoggerHook', init_kwargs=init_kwargs))

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority:
    # config file > default (base filename)
    if cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if not hasattr(cfg, 'dist_params'):
        cfg.dist_params = dict(backend='nccl')

    init_dist(args.launcher, **cfg.dist_params)
    rank, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)

    auto_resume = cfg.get('auto_resume', True)
    if auto_resume and cfg.get('resume_from', None) is None:
        resume_pth = osp.join(cfg.work_dir, 'latest.pth')
        if osp.exists(resume_pth):
            cfg.resume_from = resume_pth

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.get('log_level', 'INFO'))

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Config: {cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)

    cfg.seed = seed
    meta['seed'] = seed
    meta['config_name'] = osp.basename(args.config)
    meta['work_dir'] = osp.basename(cfg.work_dir.rstrip('/\\'))
    
    model = build_model(cfg.model)
    if dv(torch.__version__) >= dv('2.0.0') and args.compile:
        model = torch.compile(model)
    datasets = [build_dataset(cfg.data.train)]
    cfg.workflow = cfg.get('workflow', [('train', 1)])
    assert len(cfg.workflow) == 1
    if cfg.checkpoint_config is not None:
        # save protogcn version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            protogcn_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text)
    test_option = dict(test_last=args.test_last, test_best=args.test_best)

    default_mc_cfg = ('localhost', 22077)
    memcached = cfg.get('memcached', False)

    if rank == 0 and memcached:
        # mc_list is a list of pickle files you want to cache in memory.
        # Basically, each pickle file is a dictionary.
        mc_cfg = cfg.get('mc_cfg', default_mc_cfg)
        assert isinstance(mc_cfg, tuple) and mc_cfg[0] == 'localhost'
        if not test_port(mc_cfg[0], mc_cfg[1]):
            mc_on(port=mc_cfg[1], launcher=args.launcher)
        retry = 3
        while not test_port(mc_cfg[0], mc_cfg[1]) and retry > 0:
            time.sleep(5)
            retry -= 1
        assert retry >= 0, 'Failed to launch memcached. '

    dist.barrier()

    train_model(model, datasets, cfg, validate=args.validate, test=test_option, timestamp=timestamp, meta=meta)
    dist.barrier()

    if rank == 0 and memcached:
        mc_off()


if __name__ == '__main__':
    main()
