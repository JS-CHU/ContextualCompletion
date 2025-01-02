import os
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_

from diffusion_pc.utils.misc import *
from diffusion_pc.utils.data import *
from diffusion_pc.models.autoencoder import *
from dataset.vipc import vipc_valset, vipc_trainset

# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--latent_dim', type=int, default=512)
parser.add_argument('--num_steps', type=int, default=500)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--resume', type=str, default=None)

# Datasets and loaders
parser.add_argument('--cls_list', type=list, default=['02691156', '02933112', '02958343', '03001627', '03636649', '04256520', '04379243', '04530566'])
# parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--scale_mode', type=str, default=None)
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=32)
parser.add_argument('--rotate', type=eval, default=False, choices=[True, False])

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=150 * THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=500 * THOUSAND)
# Training
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_results/logs_DCG/vipc')
parser.add_argument('--dataset_path', type=str, default='ShapeNetViPC')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--max_iters', type=int, default=600 * THOUSAND)
parser.add_argument('--val_freq', type=float, default=1000)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--npoints', type=int, default=2048)
parser.add_argument('--num_inspect_batches', type=int, default=1)
parser.add_argument('--num_inspect_pointclouds', type=int, default=4)
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = os.path.join(args.log_root, 'step=500_latent=512')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
logger.info('Loading datasets...')
# train_dset = ShapeNetCore(
#     path=args.dataset_path,
#     cates=args.categories,
#     split='train',
#     scale_mode=args.scale_mode,
#     transform=transform,
# )
train_dset = vipc_trainset(missing_mode=False, root=args.dataset_path, cls_list=args.cls_list)
# val_dset = vipc_plane_valset(missing_mode=False, root=args.dataset_path, cls_list=args.cls_list)
train_iter = get_data_iterator(DataLoader(
    train_dset,
    batch_size=args.train_batch_size,
    num_workers=1,
    shuffle=True
))
# val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, num_workers=0)

# Model
logger.info('Building model...')
if args.resume is not None:
    logger.info('Resuming from checkpoint...')
    ckpt = torch.load(args.resume)
    model = AutoEncoder(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
else:
    model = AutoEncoder(args).to(args.device)
logger.info(repr(model))

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay
                             )
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)


# Train, validate
def train(it):
    # Load dataset
    batch = next(train_iter)
    gt = batch['gt'].to(args.device)
    x = batch['part'].to(args.device)

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    loss = model.get_loss(x, gt)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f ' % (it, loss.item(), orig_grad_norm))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()


# Main loop
logger.info('Start training...')
try:
    it = 1
    while it <= args.max_iters:
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, 0, opt_states, step=it)
        it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')
