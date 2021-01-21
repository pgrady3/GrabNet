import sys

sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import os
import argparse
import random

import mano
from psbody.mesh import MeshViewers, Mesh
from grabnet.tools.vis_tools import points_to_spheres
from grabnet.tools.utils import euler
from grabnet.tools.cfg_parser import Config
from grabnet.tests.tester import Tester

from bps_torch.bps import bps_torch

from psbody.mesh.colors import name_to_rgb
from grabnet.tools.train_tools import point2point_signed
from grabnet.tools.utils import aa2rotmat, rotmat2aa
from grabnet.tools.utils import makepath
from grabnet.tools.utils import to_cpu

import grabnet.tests.util as util
import sys
sys.path.append('grabnet/tests')    # Hack to make the pickle load
import grabnet.tests.hand_object as hand_object
import pickle
from tqdm import tqdm


# def convert_mano_format(ho, rh_model, rh_model_pkl):
#     optimized_pose = {}
#
#     target_verts = torch.Tensor(ho.hand_verts).unsqueeze(0)
#     mTc = torch.Tensor(ho.hand_mTc)
#     betas = torch.Tensor(ho.hand_beta).unsqueeze(0)
#     approx_trans = mTc[:3, 3].unsqueeze(0)
#     rot_mTc = mTc[:3, :3].unsqueeze(0)
#     rot_aa_pose = torch.Tensor(ho.hand_pose[:3]).unsqueeze(0).unsqueeze(0).unsqueeze(0)     # Somehow this wants a Bx1x1x3 input
#     rot_pose = aa2rotmat(rot_aa_pose).squeeze(1).squeeze(1).view(-1, 3, 3)
#
#     # rot_combined = torch.bmm(rot_pose, rot_mTc)
#     rot_combined = torch.bmm(rot_mTc, rot_pose)
#     approx_global_orient = rotmat2aa(rot_combined).squeeze(1).squeeze(1)
#
#     hand_pose_15 = torch.Tensor(ho.hand_pose[3:]).unsqueeze(0)  # Get 15-dim pca
#     mano_out_1 = rh_model_pkl(global_orient=approx_global_orient, hand_pose=hand_pose_15, betas=betas, return_full_pose=True)
#     hand_pose_in = mano_out_1.full_pose[:, 3:]     # Get 45-dim full axang representation
#     approx_trans = target_verts[:, 0, :] - mano_out_1.vertices[:, 0, :]
#
#     # if torch.isnan(approx_global_orient).any():  # Using honnotate?
#     #     approx_global_orient = torch.Tensor(ho.hand_pose[:3]).unsqueeze(0)
#
#     pose, trans, rot = util.opt_hand(rh_model, target_verts, hand_pose_in, approx_trans, approx_global_orient, betas)
#
#     print('pose before', hand_pose_in)
#     print('pose after', pose)
#     print('rot before', approx_global_orient)
#     print('rot after', rot)
#
#     optimized_pose['transl'] = trans
#     optimized_pose['global_orient'] = rot
#     optimized_pose['hand_pose'] = pose
#     optimized_pose['betas'] = betas
#
#     verts_rh_gen_cnet = rh_model(**optimized_pose).vertices
#
#     print('Hand fitting err', np.linalg.norm(verts_rh_gen_cnet.squeeze().detach().numpy() - ho.hand_verts, 2, 1).mean())
#
#     return optimized_pose


def grab_new_objs(pkl_path, mano_path):
    rh_model = mano.load(model_path=mano_path,
                         model_type='mano',
                         num_pca_comps=45,
                         batch_size=1,
                         flat_hand_mean=True)

    rh_model_pkl = mano.load(model_path=mano_path,
                             model_type='mano',
                             num_pca_comps=15,
                             batch_size=1,
                             flat_hand_mean=False)

    all_samples = pickle.load(open(pkl_path, 'rb'))

    for idx, new_obj in enumerate(tqdm(all_samples)):
        print('idx', idx)
        ho = new_obj['ho_aug']

        # obj_centroid = ho.obj_verts.mean(0)
        # ho.obj_verts = np.array(ho.obj_verts) - obj_centroid
        # ho.hand_verts = np.array(ho.hand_verts) - obj_centroid
        # ho.hand_mTc = np.array(ho.hand_mTc)
        # ho.hand_mTc[:3, 3] = ho.hand_mTc[:3, 3] - obj_centroid

        opt_dict = util.convert_pca15_aa45(ho, mano_model_in=rh_model_pkl, mano_model_out=rh_model)

    # out_file = 'datasets/{}_opt.pkl'.format(os.path.basename(pkl_path))
    # print('Saving to {}. Len {}'.format(out_file, len(all_samples)))
    # pickle.dump(all_samples, open(out_file, 'wb'))


if __name__ == '__main__':
    pkl_path = '/home/patrick/pose/align_hands/dataset/test.pkl'
    # pkl_path = '/home/patrick/pose/align_hands/dataset/im.pkl'
    print('Loading dataset', pkl_path)

    ho = hand_object.HandObject()

    mano_path = '.'

    grab_new_objs(pkl_path, mano_path)
