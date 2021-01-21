from torch.utils.data import Dataset
# from util import *
import torch
import numpy as np
from pytorch3d.structures import Meshes
import pytorch3d
from torch.utils.data import DataLoader
# import create_dataset
import time
from tqdm import tqdm
import pickle
import sys
sys.path.append('grabnet/tests')    # Hack to make the pickle load
import grabnet.tests.hand_object as hand_object

import mano
from grabnet.tests.util import convert_pca15_aa45


class ContactDBDataset(Dataset):
    def __init__(self, data_dir, train=False, min_num_cont=1):
        start_time = time.time()
        self.dataset = pickle.load(open(data_dir, 'rb'))  # Expensive step, can take up to 5 sec
        self.train = train
        self.aug_vert_jitter = 0.0005    # TODO, value?

        if 'num_verts_in_contact' in self.dataset[0]:
            print('Cutting samples less than {}. Was size {}'.format(min_num_cont, len(self.dataset)))
            self.dataset = [s for s in self.dataset if s['num_verts_in_contact'] >= min_num_cont]

        mano_path = '.'
        self.mano_model_out = mano.load(model_path=mano_path,
                             model_type='mano',
                             num_pca_comps=45,
                             batch_size=1,
                             flat_hand_mean=True)

        self.mano_model_in = mano.load(model_path=mano_path,
                                 model_type='mano',
                                 num_pca_comps=15,
                                 batch_size=1,
                                 flat_hand_mean=False)

        print('Dataset loaded in {:.2f} sec, {} samples'.format(time.time() - start_time, len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        out = dict()
        # out['obj_faces'] = torch.Tensor(sample['ho_gt'].obj_faces)
        out['obj_sampled_idx'] = torch.Tensor(sample['obj_sampled_idx']).long()

        # out['obj_verts_gt'] = torch.Tensor(sample['ho_gt'].obj_verts)
        obj_verts_gt = torch.Tensor(sample['ho_gt'].obj_verts)
        out['obj_sampled_verts_gt'] = obj_verts_gt[out['obj_sampled_idx'], :]
        # out['obj_contact_gt'] = torch.Tensor(sample['ho_gt'].obj_contact)
        # out['hand_contact_gt'] = torch.Tensor(sample['ho_gt'].hand_contact)
        # out['hand_pose_gt'] = torch.Tensor(sample['ho_gt'].hand_pose)
        out['hand_beta_gt'] = torch.Tensor(sample['ho_gt'].hand_beta)
        # out['hand_mTc_gt'] = torch.Tensor(sample['ho_gt'].hand_mTc)
        out['hand_verts_gt'] = torch.Tensor(sample['ho_gt'].hand_verts)

        # out['obj_verts_aug'] = torch.Tensor(sample['ho_aug'].obj_verts)
        # out['obj_sampled_verts_aug'] = out['obj_verts_aug'][out['obj_sampled_idx'], :]
        out['hand_pose_aug'] = torch.Tensor(sample['ho_aug'].hand_pose)
        out['hand_beta_aug'] = torch.Tensor(sample['ho_aug'].hand_beta)
        out['hand_mTc_aug'] = torch.Tensor(sample['ho_aug'].hand_mTc)
        out['hand_verts_aug'] = torch.Tensor(sample['ho_aug'].hand_verts)

        conversion_dict = convert_pca15_aa45(sample['ho_aug'], mano_model_in=self.mano_model_in, mano_model_out=self.mano_model_out)
        out['hand_trans_aug'] = conversion_dict['transl']
        out['hand_pose_45_aug'] = conversion_dict['hand_pose']
        out['hand_rot_aug'] = conversion_dict['global_orient']

        # out['hand_feats_aug'] = torch.Tensor(sample['hand_feats_aug'])
        # out['obj_feats_aug'] = torch.Tensor(sample['obj_feats_aug'])
        # out['obj_normals_aug'] = torch.Tensor(sample['ho_aug'].obj_normals)

        # if self.train:
        #     # TODO Augmentation step, more?
        #     out['obj_sampled_verts_aug'] += torch.randn(out['obj_sampled_verts_aug'].shape) * self.aug_vert_jitter

        return out

    @staticmethod
    def collate_fn(batch):
        out = dict()
        batch_keys = batch[0].keys()
        skip_keys = ['obj_faces', 'obj_verts_gt', 'obj_contact_gt', 'obj_normals_aug', 'obj_verts_aug']    # These will be manually collated

        # For each not in skip_keys, use default torch collator
        for key in [k for k in batch_keys if k not in skip_keys]:
            out[key] = torch.utils.data._utils.collate.default_collate([d[key] for d in batch])

        # verts_gt_all = [sample['obj_verts_gt'] for sample in batch]
        # verts_aug_all = [sample['obj_verts_aug'] for sample in batch]
        # faces_all = [sample['obj_faces'] for sample in batch]
        # contact_all = [sample['obj_contact_gt'] for sample in batch]
        # obj_normals_aug_all = [sample['obj_normals_aug'] for sample in batch]

        # out['obj_contact_gt'] = pytorch3d.structures.utils.list_to_padded(contact_all, pad_value=-1)
        # out['obj_normals_aug'] = pytorch3d.structures.utils.list_to_padded(obj_normals_aug_all, pad_value=-1)

        # out['obj_verts_gt'] = pytorch3d.structures.utils.list_to_padded(verts_gt_all, pad_value=-1)
        # out['obj_verts_aug'] = pytorch3d.structures.utils.list_to_padded(verts_aug_all, pad_value=-1)
        # out['obj_faces'] = pytorch3d.structures.utils.list_to_padded(faces_all, pad_value=-1)
        # out['mesh_gt'] = Meshes(verts=verts_gt_all, faces=faces_all)    # This is slower than the above, but probably fast enough
        # out['mesh_aug'] = Meshes(verts=verts_aug_all, faces=faces_all)

        return out


# Test by loading entire dataset
if __name__ == '__main__':
    # dataset = ContactDBDataset('dataset/train.pkl')
    dataset = ContactDBDataset('dataset/im_pose_estimates.pkl')

    # TODO, dataloading is slow AF. need to fix
    dataloader = DataLoader(dataset, batch_size=16, num_workers=6, collate_fn=ContactDBDataset.collate_fn)

    start_time = time.time()
    print('start', len(dataloader))
    for idx, sample in enumerate(tqdm(dataloader)):
        pass

    print('Epoch dataload time: ', time.time() - start_time)
