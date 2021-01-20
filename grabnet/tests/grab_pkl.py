# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
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
import grabnet.tests.hand_object as hand_object
import pickle
from tqdm import tqdm


def vis_results(ho, dorig, coarse_net, refine_net, rh_model, save=False, save_dir=None, rh_model_pkl=None, vis=True):

    # with torch.no_grad():
    imw, imh = 1920, 780
    cols = len(dorig['bps_object'])
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    if vis:
        mvs = MeshViewers(window_width=imw, window_height=imh, shape=[1, cols], keepalive=True)

    # drec_cnet = coarse_net.sample_poses(dorig['bps_object'])
    #
    # for k in drec_cnet.keys():
    #     print('drec cnet', k, drec_cnet[k].shape)

    # verts_rh_gen_cnet = rh_model(**drec_cnet).vertices

    drec_cnet = {}

    hand_pose_in = torch.Tensor(ho.hand_pose[3:]).unsqueeze(0)
    mano_out_1 = rh_model_pkl(hand_pose=hand_pose_in)
    hand_pose_in = mano_out_1.hand_pose

    mTc = torch.Tensor(ho.hand_mTc)
    approx_global_orient = rotmat2aa(mTc[:3, :3].unsqueeze(0))

    if torch.isnan(approx_global_orient).any(): # Using honnotate?
        approx_global_orient = torch.Tensor(ho.hand_pose[:3]).unsqueeze(0)

    approx_global_orient = approx_global_orient.squeeze(1).squeeze(1)
    approx_trans = mTc[:3, 3].unsqueeze(0)

    target_verts = torch.Tensor(ho.hand_verts).unsqueeze(0)

    pose, trans, rot = util.opt_hand(rh_model, target_verts, hand_pose_in, approx_trans, approx_global_orient)

    # drec_cnet['hand_pose'] = torch.einsum('bi,ij->bj', [hand_pose_in, rh_model_pkl.hand_components])
    drec_cnet['transl'] = trans
    drec_cnet['global_orient'] = rot
    drec_cnet['hand_pose'] = pose

    verts_rh_gen_cnet = rh_model(**drec_cnet).vertices

    _, h2o, _ = point2point_signed(verts_rh_gen_cnet, dorig['verts_object'].to(device))

    drec_cnet['trans_rhand_f'] = drec_cnet['transl']
    drec_cnet['global_orient_rhand_rotmat_f'] = aa2rotmat(drec_cnet['global_orient']).view(-1, 3, 3)
    drec_cnet['fpose_rhand_rotmat_f'] = aa2rotmat(drec_cnet['hand_pose']).view(-1, 15, 3, 3)
    drec_cnet['verts_object'] = dorig['verts_object'].to(device)
    drec_cnet['h2o_dist']= h2o.abs()

    print('Hand fitting err', np.linalg.norm(verts_rh_gen_cnet.squeeze().detach().numpy() - ho.hand_verts, 2, 1).mean())
    orig_obj = dorig['mesh_object'][0].v
    # print(orig_obj.shape, orig_obj)
    # print('Obj fitting err', np.linalg.norm(orig_obj - ho.obj_verts, 2, 1).mean())


    drec_rnet = refine_net(**drec_cnet)
    mano_out = rh_model(**drec_rnet)
    verts_rh_gen_rnet = mano_out.vertices
    joints_out = mano_out.joints

    if vis:
        for cId in range(0, len(dorig['bps_object'])):
            try:
                from copy import deepcopy
                meshes = deepcopy(dorig['mesh_object'])
                obj_mesh = meshes[cId]
            except:
                obj_mesh = points_to_spheres(to_cpu(dorig['verts_object'][cId]), radius=0.002, vc=name_to_rgb['green'])

            hand_mesh_gen_cnet = Mesh(v=to_cpu(verts_rh_gen_cnet[cId]), f=rh_model.faces, vc=name_to_rgb['pink'])
            hand_mesh_gen_rnet = Mesh(v=to_cpu(verts_rh_gen_rnet[cId]), f=rh_model.faces, vc=name_to_rgb['gray'])

            if 'rotmat' in dorig:
                rotmat = dorig['rotmat'][cId].T
                obj_mesh = obj_mesh.rotate_vertices(rotmat)
                hand_mesh_gen_cnet.rotate_vertices(rotmat)
                hand_mesh_gen_rnet.rotate_vertices(rotmat)
                # print('rotmat', rotmat)

            hand_mesh_gen_cnet.reset_face_normals()
            hand_mesh_gen_rnet.reset_face_normals()

            # mvs[0][cId].set_static_meshes([hand_mesh_gen_cnet] + obj_mesh, blocking=True)
            # mvs[0][cId].set_static_meshes([hand_mesh_gen_rnet,obj_mesh], blocking=True)
            mvs[0][cId].set_static_meshes([hand_mesh_gen_rnet,hand_mesh_gen_cnet,obj_mesh], blocking=True)

            if save:
                save_path = os.path.join(save_dir, str(cId))
                makepath(save_path)
                hand_mesh_gen_rnet.write_ply(filename=save_path + '/rh_mesh_gen_%d.ply' % cId)
                obj_mesh[0].write_ply(filename=save_path + '/obj_mesh_%d.ply' % cId)

    return verts_rh_gen_rnet, joints_out


def grab_new_objs(grabnet, pkl_path, rot=True, n_samples=5, scale=1.):
    grabnet.coarse_net.eval()
    grabnet.refine_net.eval()

    rh_model = mano.load(model_path=grabnet.cfg.rhm_path,
                         model_type='mano',
                         num_pca_comps=45,
                         batch_size=n_samples,
                         flat_hand_mean=True).to(grabnet.device)

    rh_model_pkl = mano.load(model_path=grabnet.cfg.rhm_path,
                         model_type='mano',
                         num_pca_comps=15,
                         batch_size=n_samples,
                         flat_hand_mean=False).to(grabnet.device)

    grabnet.refine_net.rhm_train = rh_model

    grabnet.logger(f'################# \n'
                   f'Colors Guide:'
                   f'                   \n'
                   f'Gray  --->  GrabNet generated grasp\n')

    bps = bps_torch(custom_basis = grabnet.bps)

    all_samples = pickle.load(open(pkl_path, 'rb'))

    if args.vis:
        print('Shuffling!!!')
        random.shuffle(all_samples)

    all_samples = all_samples[:args.num]
    all_data = []

    for idx, new_obj in enumerate(tqdm(all_samples)):
        print('idx', idx)
        ho = new_obj['ho_aug']

        obj_centroid = ho.obj_verts.mean(0)
        ho.obj_verts = np.array(ho.obj_verts) - obj_centroid
        ho.hand_verts = np.array(ho.hand_verts) - obj_centroid
        ho.hand_mTc = np.array(ho.hand_mTc)
        ho.hand_mTc[:3, 3] = ho.hand_mTc[:3, 3] - obj_centroid



        rand_rotdeg = np.random.random([n_samples, 3]) * np.array([0, 0, 0])

        rand_rotmat = euler(rand_rotdeg)
        dorig = {'bps_object': [],
                 'verts_object': [],
                 'mesh_object': [],
                 'rotmat':[]}

        for samples in range(n_samples):

            verts_obj, mesh_obj, rotmat = load_obj_verts(ho, rand_rotmat[samples], rndrotate=rot, scale=scale)
            
            bps_object = bps.encode(verts_obj, feature_type='dists')['dists']

            dorig['bps_object'].append(bps_object.to(grabnet.device))
            dorig['verts_object'].append(torch.from_numpy(verts_obj.astype(np.float32)).unsqueeze(0))
            dorig['mesh_object'].append(mesh_obj)
            dorig['rotmat'].append(rotmat)
            obj_name = 'test1'

        dorig['bps_object'] = torch.cat(dorig['bps_object'])
        dorig['verts_object'] = torch.cat(dorig['verts_object'])

        save_dir = os.path.join(grabnet.cfg.work_dir, 'grab_new_objects')
        # grabnet.logger(f'#################\n'
        #                       f'                   \n'
        #                       f'Showing results for the {obj_name.upper()}'
        #                       f'                      \n')

        verts_out, joints_out = vis_results(ho, dorig=dorig,
                    coarse_net=grabnet.coarse_net,
                    refine_net=grabnet.refine_net,
                    rh_model=rh_model,
                    save=False,
                    save_dir=save_dir,
                    rh_model_pkl=rh_model_pkl, vis=args.vis
                    )

        ho.obj_verts = np.array(ho.obj_verts) + obj_centroid
        ho.hand_verts = np.array(ho.hand_verts) + obj_centroid
        ho.hand_mTc = np.array(ho.hand_mTc)
        ho.hand_mTc[:3, 3] = ho.hand_mTc[:3, 3] + obj_centroid

        verts_out = np.array(verts_out.detach().squeeze().numpy()) + obj_centroid
        joints_out = np.array(joints_out.detach().squeeze().numpy()) + obj_centroid

        new_ho = hand_object.HandObject()
        new_ho.load_from_verts(verts_out, new_obj['ho_gt'].obj_faces, new_obj['ho_gt'].obj_verts)
        all_data.append({'gt_ho': new_obj['ho_gt'], 'in_ho': new_obj['ho_aug'], 'out_verts': verts_out, 'out_joints': joints_out})

    out_file = 'fitted_grabnet.pkl'
    print('Saving to {}. Len {}'.format(out_file, len(all_data)))
    pickle.dump(all_data, open(out_file, 'wb'))


def load_obj_verts(ho, rand_rotmat, rndrotate=True, scale=1., n_sample_verts=10000):
    np.random.seed(100)

    obj_mesh = Mesh(v=ho.obj_verts, f=ho.obj_faces, vscale=scale)
    # obj_mesh = Mesh(filename=mesh_path, vscale=scale)

    obj_mesh.reset_normals()
    obj_mesh.vc = obj_mesh.colors_like('green')

    ## center and scale the object
    # max_length = np.linalg.norm(obj_mesh.v, axis=1).max()
    # if  max_length > .3:
    #     re_scale = max_length/.08
    #     print(f'The object is very large, down-scaling by {re_scale} factor')
    #     obj_mesh.v = obj_mesh.v/re_scale
    #
    object_fullpts = obj_mesh.v
    # maximum = object_fullpts.max(0, keepdims=True)
    # minimum = object_fullpts.min(0, keepdims=True)
    #
    # offset = ( maximum + minimum) / 2
    verts_obj = object_fullpts # - offset
    # obj_mesh.v = verts_obj

    # if rndrotate:
    #     obj_mesh.rotate_vertices(rand_rotmat)
    #     verts_obj = obj_mesh.v
    
    if verts_obj.shape[0] > n_sample_verts:
        verts_sample_id = np.random.choice(verts_obj.shape[0], n_sample_verts, replace=False)
    else:
        verts_sample_id = np.arange(verts_obj.shape[0])
    verts_sampled = verts_obj[verts_sample_id]

    return verts_sampled, obj_mesh, rand_rotmat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GrabNet-Testing')

    parser.add_argument('--rhm-path', type=str, default='.', help='The path to the folder containing MANO_RIHGT model')
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--iters', type=int, default=3)
    parser.add_argument('--vis', action='store_true')
    # pkl_path = '/home/patrick/pose/align_hands/dataset/fine.pkl'
    pkl_path = '/home/patrick/pose/align_hands/dataset/im.pkl'
    print('Loading dataset', pkl_path)

    args = parser.parse_args()

    rhm_path = args.rhm_path

    cwd = os.getcwd()
    work_dir = cwd + '/logs'

    best_cnet = 'grabnet/models/coarsenet.pt'
    best_rnet = 'grabnet/models/refinenet.pt'
    bps_dir = 'grabnet/configs/bps.npz'

    cfg_path = 'grabnet/configs/grabnet_cfg.yaml'

    config = {
        'work_dir': work_dir,
        'best_cnet': best_cnet,
        'best_rnet': best_rnet,
        'bps_dir': bps_dir,
        'rhm_path': rhm_path
    }

    cfg = Config(default_cfg_path=cfg_path, **config)

    grabnet = Tester(cfg=cfg)

    print('Setting refineNet iters', args.iters)
    grabnet.refine_net.n_iters = args.iters

    grab_new_objs(grabnet, pkl_path, rot=True, n_samples=1)
