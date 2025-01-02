import argparse
from models.RestoreNet import RestoreNet, RestoreNet_rotate_back, RestoreNet_rotate_back_similar_MLP, RestoreNet_rotate_back_similar_gate
from dataset.vipc import vipc_trainset
from torch.utils.data import DataLoader
import torch.utils.tensorboard
from utils.loss import CD, CD_sqrt
from utils.diffusion_func import *
from utils.logger import get_logger
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '8387'


def train(gpu, args):
    if gpu == 1:
        time.sleep(0.5)
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        world_size=args.world_size,
        rank=rank
    )
    args.device=torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')


    if not os.path.exists(args.restore_log_dir):
        os.makedirs(args.restore_log_dir)

    logger = get_logger('train', args.restore_log_dir)
    logger.info(args)

    logger.info('Building model...')
    net = RestoreNet_rotate_back_similar_gate(args)
    if rank == 0 and args.model_path:
        logger.info('Loading model...')
        net.load_state_dict(torch.load(args.model_path))
    net.to(args.device)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu])
    # net = nn.DataParallel(net, [0, 1])
    net.train()
    logger.info(repr(net))

    writer = torch.utils.tensorboard.SummaryWriter(args.restore_log_dir)
    logger.info('Loading datasets...')
    train_dataset = vipc_trainset(root=args.train_root, cls_list=args.cls_list, missing_mode=False)
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=False, sampler=train_sampler, num_workers=args.train_num_workers)
    steps = len(train_dataloader)
    # print(steps)

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=args.train_lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.sche_step, gamma=args.sche_gamma)

    for epoch in range(args.start_epoch, args.nepoch):
        loss_list = []
        train_sampler.set_epoch(epoch)
        for batch_id, data in enumerate(train_dataloader):
            # load gt
            gt = data['gt'].to(args.device)
            # true_missing_mask = data['missing_mask'].to(args.device)

            # reset optimizer
            optimizer.zero_grad()

            P = data['part'].to(args.device)
            # normals = data['part_normals'].to(args.device)

            # generate Q
            # Q = diffusion_gen(pc=P, sample_num_points=args.base_points, normalize=args.scale_mode, d_ckpt=args.d_ckpt, e_ckpt=args.e_ckpt, device=args.device)
            Q = AE(pc=P, sample_num_points=args.base_points, ckpt=args.e_ckpt, device=args.device)
            # save diffusion gen
            # np.save(os.path.join(args.diffusion_log_dir, 'diffusion_gen_' + str(batch_id) + '.npy'), Q.cpu().numpy())

            # restore
            _, q_refine, _, _, _ = net(P, Q)

            # get_loss
            # mirroring_loss = 100 * CD_loss(new_p, gt)
            refine_loss = 100 * CD(q_refine, gt)

            # loss = (refine_loss + restore_loss) / 2
            loss = refine_loss

            writer.add_scalar('train/loss', loss, batch_id + epoch * steps)
            writer.add_scalar('train/refine_loss', refine_loss, batch_id + epoch * steps)
            # logger.info('[Train] Epoch %03d Batch %03d | Loss %.6f RestoreLoss %.6f MissingLoss %.6f' % (epoch, batch_id, loss.item(), restore_loss.item(), missing_loss.item()))
            loss_list.append(loss.item())

            loss.backward()
            orig_grad_norm = clip_grad_norm_(net.parameters(), args.max_grad_norm)
            optimizer.step()
            logger.info('[Train] Epoch %03d Batch %03d | RefineLoss %.6f | Grad %.4f' % (epoch, batch_id, refine_loss.item(), orig_grad_norm))
            # for name, param in net.named_parameters():
            #     if param.grad is not None:
            #         print(f"Parameter: {name}, Gradient: {param.grad}")
            #     else:
            #         print(f"Parameter: {name}, Gradient: None")

        scheduler.step()  # 学习率迭代次数+1
        avg_loss = sum(loss_list) / len(loss_list)
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('train/lr', current_lr, epoch)
        writer.add_scalar('train/avg_loss', avg_loss, epoch)
        logger.info('[Train] Epoch %03d |lr %.6f, Loss %.6f' % (epoch, current_lr, avg_loss))

        if rank == 0 and epoch % args.save_interval == 0:
            torch.save(net.module.state_dict(), args.restore_log_dir + "/model_{}.pth".format(epoch))
            # torch.save(net.state_dict(), args.restore_log_dir + "/model_{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DDP settings
    # 节点数/主机数
    parser.add_argument('-n', '--nodes', default=1, type=int, help='the number of nodes/computer')
    # 一个节点/主机上面的GPU数
    parser.add_argument('-g', '--gpus', default=2, type=int, help='the number of gpus per nodes')
    # 当前主机的编号，例如对于n机m卡训练，则nr∈[0,n-1]。对于单机多卡，nr只需为0。
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--world_size', type=int, default=2)

    # sche settings
    parser.add_argument('--sche_step', type=int, default=5)
    parser.add_argument('--sche_gamma', type=int, default=0.8)

    # train settings
    parser.add_argument('--train_root', type=str, default='./logs_results/logs_CRef/vipc')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--cls_list', type=list, default=['02691156', '02933112', '02958343', '03001627', '03636649', '04256520', '04379243', '04530566']) # test 8 vs 13
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--restore_log_dir', type=str, default='./logs_results/results_CRef/vipc')
    parser.add_argument('--k', type=int, default=64)
    parser.add_argument('--similar_num', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--fdims', type=int, default=448, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--global_fdims', type=int, default=512, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--train_num_workers', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=float, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--nepoch', type=int, default=51)
    parser.add_argument('--base_points', type=int, default=2048)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--e_ckpt', type=str, default='./logs_results/logs_DCG/vipc/step=500_latent=512/600000.ckpt')
    args = parser.parse_args()
    # args.model_path = None

    mp.spawn(
        train,
        nprocs=args.gpus,
        args=(args,)
    )
