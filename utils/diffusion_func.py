from diffusion_pc.models.autoencoder import *
from diffusion_pc.utils.misc import *
from diffusion_pc.models.vae_gaussian import *
from diffusion_pc.models.vae_flow import *
import os
import numpy as np
from utils.fps import farthest_point_sample_np
import torch
from utils.rotate import rotate_pc
from utils.normalize import normalize_point_clouds
import pickle


def val_encoder(pc, ckpt, device):

    # Checkpoint
    ckpt = torch.load(ckpt)
    seed_all(ckpt['args'].seed)

    # Model
    model = AutoEncoder(ckpt['args']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    with torch.no_grad():
        code = model.encode(pc)
    return code


def val_AE(pc_path, save_dir, ckpt, device):

    # Logging
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = get_logger('test', save_dir)

    # Checkpoint
    ckpt = torch.load(ckpt)
    seed_all(ckpt['args'].seed)

    # Datasets and loaders
    logger.info('Loading dataset...')
    test_ref = np.loadtxt(pc_path)
    test_pc = farthest_point_sample_np(test_ref, 1000)

    test_ref = rotate_pc(torch.FloatTensor(test_ref))
    test_pc = rotate_pc(torch.FloatTensor(test_pc))

    test_ref = test_ref.view(1, test_ref.shape[0], test_ref.shape[1])
    test_pc = test_pc.view(1, test_pc.shape[0], test_pc.shape[1])

    pc, _, _ = normalize_point_clouds(test_pc, mode='shape_bbox').to(device)
    test_ref, _, _ = normalize_point_clouds(test_ref, mode='shape_bbox').to(device)

    # Model
    logger.info('Loading model...')
    model = AutoEncoder(ckpt['args']).to(device)
    model.load_state_dict(ckpt['state_dict'])

    all_ref = []
    all_recons = []

    model.eval()
    with torch.no_grad():
        code = model.encode(pc)
        recons = model.decode(code, pc.shape[1], flexibility=ckpt['args'].flexibility).detach()

    all_ref.append(test_ref.detach().cpu())
    all_recons.append(recons.detach().cpu())

    all_ref = torch.cat(all_ref, dim=0)
    all_recons = torch.cat(all_recons, dim=0)

    logger.info('Saving point clouds...')
    np.savetxt(os.path.join(save_dir, 'ae_ref.xyz'), all_ref[0].numpy())
    np.savetxt(os.path.join(save_dir, 'ae_out.xyz'), all_recons[0].numpy())


def val_gen(pc_path, vipc, save_dir, sample_num_points, normalize, ckpt, device):
    seed = 9988
    batch_size = 1

    # Logging
    save_dir = os.path.join(save_dir, 'diffusion_save')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = get_logger('test', save_dir)

    # Checkpoint
    ckpt = torch.load(ckpt)
    seed_all(seed)

    # Datasets and loaders
    logger.info('Loading datasets...')

    if vipc:
        with open(pc_path, 'rb') as f:
            test_ref = pickle.load(f).astype(np.float32)
    else:
        test_ref = np.loadtxt(pc_path)

    test_pc = farthest_point_sample_np(test_ref, 1000)

    test_ref = rotate_pc(torch.FloatTensor(test_ref))
    test_pc = rotate_pc(torch.FloatTensor(test_pc))

    test_ref = test_ref.view(1, test_ref.shape[0], test_ref.shape[1])
    test_pc = test_pc.view(1, test_pc.shape[0], test_pc.shape[1])

    pc, _, _ = normalize_point_clouds(test_pc, mode='shape_bbox').to(device)
    test_ref, _, _ = normalize_point_clouds(test_ref, mode='shape_bbox').to(device)

    # Model
    logger.info('Loading model...')
    if ckpt['args'].model == 'gaussian':
        model = GaussianVAE(ckpt['args']).to(device)
    elif ckpt['args'].model == 'flow':
        model = FlowVAE(ckpt['args']).to(device)
    # if ckpt['args'].spectral_norm:
    #     add_spectral_norm(model, logger=logger)
    model.load_state_dict(ckpt['state_dict'])
    # Reference Point Clouds

    # Generate Point Clouds
    gen_pcs = []
    with torch.no_grad():
        # z = torch.randn([batch_size, ckpt['args'].latent_dim]).to(device)
        z = val_encoder(pc, ckpt, device).to(device)
        x = model.sample(z, sample_num_points, flexibility=ckpt['args'].flexibility)
        gen_pcs.append(x.detach().cpu())
    if normalize is not None:
        gen_pcs, _, _ = normalize_point_clouds(gen_pcs[0], mode=normalize)

    return gen_pcs


def diffusion_gen(pc, sample_num_points, d_ckpt, e_ckpt, normalize=None, device='cuda'):
    seed = 2024
    # batch_size = 1

    # Checkpoint
    ckpt = torch.load(d_ckpt)
    seed_all(seed)

    if ckpt['args'].model == 'flow':
        model = FlowVAE(ckpt['args']).to(device)
    elif ckpt['args'].model == 'gaussian':
        model = GaussianVAE(ckpt['args']).to(device)
    else:
        model =  GaussianVAE(ckpt['args']).to(device)

    model.load_state_dict(ckpt['state_dict'])
    # model = nn.DataParallel(model, [0, 1])
    model.eval()
    # Generate Point Clouds

    with torch.no_grad():
        # z = torch.randn([batch_size, ckpt['args'].latent_dim]).to(device)
        # z = val_encoder(pc, e_ckpt, device).to(device)
        z_mu, z_sigma = model.encoder(pc)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)
        gen_pc = model.diffusion.sample(sample_num_points, z, flexibility=ckpt['args'].flexibility)
        if normalize is not None:
            gen_pc, _, _ = normalize_point_clouds(gen_pc, mode=normalize)

    return gen_pc


def AE_code(pc, taxonomy_code, sample_num_points, ckpt, device):
    # Checkpoint
    ckpt = torch.load(ckpt)
    seed_all(ckpt['args'].seed)
    # print(ckpt['args'])
    model = AutoEncoder_cls_code(ckpt['args']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Generate Point Clouds
    with torch.no_grad():
        code = model.encode(pc)
        taxonomy_code = torch.FloatTensor(taxonomy_code).view(code.size(0), -1).to(code.device)
        code = torch.cat((code, taxonomy_code), dim=-1)
        recons = model.decode(code, sample_num_points, flexibility=ckpt['args'].flexibility)

    return recons

def AE(pc, sample_num_points, ckpt, device):
    # Checkpoint
    ckpt = torch.load(ckpt)
    seed_all(ckpt['args'].seed)
    # print(ckpt['args'])
    model = AutoEncoder(ckpt['args']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Generate Point Clouds
    with torch.no_grad():
        code = model.encode(pc)
        recons = model.decode(code, sample_num_points, flexibility=ckpt['args'].flexibility)

    return recons
