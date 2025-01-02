from torch.utils.data import DataLoader
from dataset.vipc import vipc_valset
from models.RestoreNet import RestoreNet, RestoreNet_rotate_back, RestoreNet_rotate_back_similar_MLP, RestoreNet_rotate_back_similar_gate
import torch.utils.tensorboard
from utils.metrics import CD
from utils.metrics import F2_score as Fscore
from utils.diffusion_func import *
from utils.logger import get_logger
import argparse
import time

def val_restore(args):

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    logger = get_logger('test', args.save_dir)
    logger.info(args)

    net = RestoreNet_rotate_back_similar_gate(args)
    net.load_state_dict(torch.load(args.model_path))
    net.to(args.device)
    net.eval()

    # load dataset
    logger.info("Loading dataset...")
    test_dataset = vipc_valset(root=args.val_root, cls_list=args.cls_list, missing_mode=False)
    test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=True, num_workers=args.val_num_workers)

    refine_cd_l2_list = []
    gen_cd_l2_list = []
    # gen_emd_list = []
    # refine_emd_list = []
    refine_Fscore_list = []

    min_cd = 100
    # print(args.e_ckpt['args'])
    cd_list = [1000,1000,1000,1000,1000,1000,1000,1000]
    file_list = ['', '', '', '', '', '', '', '']
    dict = {
        '02691156':0,
        '02933112':1,
        '02958343':2,
        '03001627':3,
        '03636649':4,
        '04256520':5,
        '04379243':6,
        '04530566':7
    }
    with torch.no_grad():
        for batch_id, data in enumerate(test_loader):
            gt = data['gt'].to(args.device)
            P = data['part'].to(args.device)
            batch_size = gt.shape[0]
            file = data['file']
            name = data['name']
            # true_missing_mask = data['missing_mask'].to(args.device)

            # generate Q
            # Q = diffusion_gen(pc=P, sample_num_points=args.base_points, normalize=args.scale_mode, d_ckpt=args.d_ckpt, e_ckpt=args.e_ckpt, device=args.device)
            start_total = time.time()
            start_DCG = time.time()
            Q = AE(pc=P, sample_num_points=args.base_points, ckpt=args.e_ckpt, device=args.device)
            # save_cls_path = os.path.join(args.save_dir, cls)
            end_DCG = time.time()
            execution_DCG = end_DCG - start_DCG
            print("DCG: " + str(execution_DCG) + 's')

            # restore
            start_CRef = time.time()
            q_cat1, q_cat, refine, _ = net(P, Q)
            end_CRef = time.time()
            execution_CRef = end_CRef - start_CRef
            print("CRef: " + str(execution_CRef) + 's')
            end_total = time.time()
            execution_total = end_total - start_total
            print("Total: " + str(execution_total) + 's')

            gen_cd_l2 = 1000 * CD(Q, gt)
            refine_cd_l2 = 1000 * CD(refine, gt)
            # gen_emd = EMD_loss(Q, gt)
            # refine_emd = EMD_loss(refine, gt)
            refine_Fscore = Fscore(refine, gt, th=0.001)

            if refine_cd_l2 < min_cd:
                min_cd = refine_cd_l2
                min_files = file

            for i in range(batch_size):
                # true_missing = gt[i][true_missing_mask[i], :]
                # cls = data['name']
                save_path = os.path.join(args.save_dir, file[i])
                if not os.path.isdir(save_path[:-3]):
                    # print(save_path[:-3])
                    os.makedirs(save_path[:-3])

                # refine_cd_l2 = 1000 * CD(refine[i].unsqueeze(0), gt[i].unsqueeze(0))
                # if refine_cd_l2 < cd_list[dict[name[i]]]:
                #     cd_list[dict[name[i]]] = refine_cd_l2.item()
                #     file_list[dict[name[i]]] = file[i]

                np.savetxt(save_path + '_gen.xyz', Q[i].cpu().numpy())
                np.savetxt(save_path + '.xyz', P[i].cpu().numpy())
                np.savetxt(save_path + '_q_cat1.xyz', q_cat1[i].cpu().numpy())
                np.savetxt(save_path + '_q_cat.xyz', q_cat[i].cpu().numpy())
                np.savetxt(save_path + '_gt.xyz', gt[i].cpu().numpy())
                np.savetxt(save_path + '_refine.xyz', refine[i].cpu().numpy())

            logger.info('[Val] Batch %03d | Gen_CD_l2 %.6f Refine_CD_l2 %.6f Refine F-Score %.6f'
                        % (batch_id, gen_cd_l2.item(), refine_cd_l2.item(), refine_Fscore))
            refine_cd_l2_list.append(refine_cd_l2.item())
            gen_cd_l2_list.append(gen_cd_l2.item())
            # gen_emd_list.append(gen_emd.item())
            # refine_emd_list.append(refine_emd.item())
            refine_Fscore_list.append(refine_Fscore)

        avg_gen_cd_l2 = sum(gen_cd_l2_list) / len(gen_cd_l2_list)
        avg_refine_cd_l2 = sum(refine_cd_l2_list) / len(refine_cd_l2_list)
        # avg_gen_emd = sum(gen_emd_list) / len(gen_emd_list)
        # avg_refine_emd = sum(refine_emd_list) / len(refine_emd_list)
        avg_refine_Fscore = sum(refine_Fscore_list) / len(refine_Fscore_list)
        logger.info('[Val] AvgGenCD_l2 %.6f AvgRefineCD_l2 %.6f AvgRefineF-Score %.6f' % (avg_gen_cd_l2, avg_refine_cd_l2, avg_refine_Fscore))
        logger.info('Min CD files %s |' % min_files)
        print(cd_list)
        print(file_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_root', type=str, default='ShapeNetViPC')
    parser.add_argument('--cls_list', type=list, default=['02691156', '02933112', '02958343', '03001627', '03636649', '04256520', '04379243', '04530566']) # ['02691156', '02933112', '02958343', '03001627', '03636649', '04256520', '04379243', '04530566'])
    parser.add_argument('--model_path', type=str, default='./logs_results/logs_CRef/model_best.pth')
    parser.add_argument('--save_dir', type=str, default='./logs_results/results_completion/vipc')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--k', type=int, default=64)
    parser.add_argument('--similar_num', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--fdims', type=int, default=448, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--global_fdims', type=int, default=512, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--val_num_workers', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--base_points', type=int, default=2048)
    parser.add_argument('--e_ckpt', type=str, default='./logs_results/logs_DCG/vipc/step=500_latent=512/600000.pt')
    args = parser.parse_args()
    val_restore(args)
