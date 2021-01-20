from os import path as osp
import numpy as np
from open3d import io as o3dio
from open3d import geometry as o3dg
from open3d import utility as o3du
from open3d import visualization as o3dv
import json
import transforms3d.quaternions as txq
import torch
import pytorch3d
from pytorch3d.structures import Meshes
import grabnet.tests.util as util
from manopth.manolayer import ManoLayer
import matplotlib.pyplot as plt


def mano_get_faces():
    return util.get_mano_closed_faces()


class HandObject:
    """Data structure to handle hand, object and contact
    Core representation: meshes"""
    closed_faces = util.get_mano_closed_faces()

    def __init__(self):
        self.is_left = None
        self.hand_beta = None
        self.hand_pose = None
        self.hand_mTc = None
        self.hand_contact = None
        self.hand_verts = None
        self.hand_joints = None
        self.obj_verts = None
        self.obj_faces = None
        self.obj_contact = None
        self.path = None

        self.obj_normals = None     # Stays None most of the time, super hacky

    def load_from_verts(self, hand_verts, obj_faces, obj_verts):
        """Load from hand/object verts alone. Incomplete data"""
        self.obj_verts = obj_verts
        self.obj_faces = obj_faces
        self.hand_verts = hand_verts

        # self.calc_dist_contact(hand=True, obj=True)

    def load_from_image(self, hand_beta, hand_pose, obj_faces, obj_verts, hand_verts=None):
        """Load from Yana Hasson's image-based method"""

        self.hand_beta = hand_beta
        self.hand_pose = hand_pose
        self.hand_mTc = np.eye(4)
        self.obj_verts = obj_verts
        self.obj_faces = obj_faces

        self.run_mano()   # Run mano model forwards
        if hand_verts is not None:
            displ = hand_verts[0, :] - self.hand_verts[0, :]
            self.hand_mTc[:3, 3] = displ
            self.run_mano()  # Rerun mano model to account for translation

            mean_err = np.linalg.norm(self.hand_verts - hand_verts, 2, 1)
            if mean_err.mean() > 1e-6:
                # print('Pose', hand_pose)
                # print('Beta', hand_pose)
                print('Mean verts error', mean_err.mean())
                print('Mano reconstruction failure')

                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(self.hand_verts[:, 0], self.hand_verts[:, 1], self.hand_verts[:, 2])
                # ax.scatter(hand_verts[:, 0], hand_verts[:, 1], hand_verts[:, 2])
                # ax.set_xlabel('X Label')
                # ax.set_ylabel('Y Label')
                # ax.set_zlabel('Z Label')
                # plt.show()

                # raise ValueError('Mano reconstruction failure')

        # self.calc_dist_contact(hand=True, obj=True)
        self.hand_contact = np.zeros((self.hand_verts.shape[0], 1)) # Set to zero since we don't know the ground truth
        self.obj_contact = np.zeros((self.obj_verts.shape[0], 1))

    def load_from_batch(self, hand_beta, hand_pose, hand_mTc, hand_contact, obj_contact, obj_mesh, idx=0, obj_rot=None):
        """For torch dataloader batch to a single HO object"""
        obj_verts = obj_mesh.verts_list()[idx]
        if obj_rot is not None:
            obj_verts = util.apply_rot(obj_rot[idx, :, :].unsqueeze(0).detach().cpu(), obj_verts.unsqueeze(0), around_centroid=True).squeeze(0)

        self.hand_beta = hand_beta[idx, :].detach().cpu().numpy()
        self.hand_pose = hand_pose[idx, :].detach().cpu().numpy()
        self.hand_mTc = hand_mTc[idx, :, :].detach().cpu().numpy()
        self.hand_contact = hand_contact[idx, :, :].detach().cpu().numpy()
        self.obj_verts = obj_verts.detach().cpu().numpy()
        self.obj_faces = obj_mesh.faces_list()[idx].detach().cpu().numpy()
        self.obj_contact = obj_contact[idx, :self.obj_verts.shape[0], :].detach().cpu().numpy()    # Since we're using a padded array, need to cut off some

        self.run_mano()

    def load_from_file(self, p_id, object_name, data_dir):
        """Load structure from original ContactPose dataset"""

        obj_filename = osp.join(data_dir, p_id, object_name, '{:s}.ply'.format(object_name))
        self.path = (p_id, object_name, data_dir)
        obj_mesh = o3dio.read_triangle_mesh(obj_filename)

        vertex_colors = np.array(obj_mesh.vertex_colors, dtype=np.float32)
        self.obj_contact = np.expand_dims(util.fit_sigmoid(vertex_colors[:, 0]), axis=1)    # Normalize with sigmoid, shape (V, 1)
        self.obj_verts = np.array(obj_mesh.vertices, dtype=np.float32)     # Keep as floats since torch uses floats
        self.obj_faces = np.array(obj_mesh.triangles)

        # Load hand
        hand_filename = osp.join(data_dir, p_id, object_name, 'mano_fits_15.json')
        with open(hand_filename, 'r') as f:
            mano_params = json.load(f)

        for idx, mp in enumerate(mano_params):
            if not mp['valid']:
                continue

            self.is_left = idx == 0  # Left then right
            self.hand_beta = np.array(mp['betas'])  # 10 shape PCA parameters
            self.hand_pose = np.array(mp['pose'])  # 18 dim length, first 3 ax-angle, 15 PCA pose

            mTc = np.eye(4)
            mTc[:3, 3] = mp['mTc']['translation']   # * 1000.0, millimeter
            mTc[:3, :3] = txq.quat2mat(mp['mTc']['rotation'])  # Object to world
            mTc = np.linalg.inv(mTc)  # World to object
            self.hand_mTc = mTc

        if self.is_left:
            raise ValueError('Pipeline currently cant handle left hands')

        self.run_mano()
        self.calc_dist_contact(hand=True, obj=False)

    def load_from_ho(self, ho, aug_pose=None, aug_trans=None):
        """Load from another HandObject obj with augmentation"""
        self.hand_beta = np.array(ho.hand_beta)
        self.hand_pose = np.array(ho.hand_pose)
        self.hand_mTc = np.array(ho.hand_mTc)
        self.obj_verts = ho.obj_verts
        self.obj_faces = ho.obj_faces
        self.obj_contact = ho.obj_contact

        if aug_pose is not None:
            self.hand_pose += aug_pose
        if aug_trans is not None:
            self.hand_mTc[:3, 3] += aug_trans

        self.run_mano()
        # self.calc_dist_contact(hand=True, obj=False)  # DONT calculate hand contact, since it's not ground truth

    def run_mano(self):
        """Runs forward_mano with numpy-pytorch-numpy handling"""
        if self.hand_pose.shape[0] == 48:   # Special case when we're loading GT honnotate
            mano_model = ManoLayer(mano_root='mano/models', joint_rot_mode="axisang", use_pca=False, center_idx=None, flat_hand_mean=True)
        else:   # Everything else
            mano_model = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=15, side='right', flat_hand_mean=False)

        pose_tensor = torch.Tensor(self.hand_pose).unsqueeze(0)
        beta_tensor = torch.Tensor(self.hand_beta).unsqueeze(0)
        tform_tensor = torch.Tensor(self.hand_mTc).unsqueeze(0)
        mano_verts, mano_joints = util.forward_mano(mano_model, pose_tensor, beta_tensor, [tform_tensor])
        self.hand_verts = mano_verts.squeeze().detach().numpy()
        self.hand_joints = mano_joints.squeeze().detach().numpy()

    def generate_pointnet_features(self, obj_sampled_idx):
        """Calculates per-point features for pointnet"""
        obj_mesh = Meshes(verts=[torch.Tensor(self.obj_verts)], faces=[torch.Tensor(self.obj_faces)])
        hand_mesh = Meshes(verts=[torch.Tensor(self.hand_verts)], faces=[torch.Tensor(util.get_mano_closed_faces())])

        obj_sampled_verts_tensor = obj_mesh.verts_padded()[:, obj_sampled_idx, :]
        _, _, obj_nearest = pytorch3d.ops.knn_points(obj_sampled_verts_tensor, hand_mesh.verts_padded(), K=1, return_nn=True)  # Calculate on object
        _, _, hand_nearest = pytorch3d.ops.knn_points(hand_mesh.verts_padded(), obj_sampled_verts_tensor, K=1, return_nn=True)  # Calculate on hand

        obj_normals = obj_mesh.verts_normals_padded()
        obj_normals = torch.nn.functional.normalize(obj_normals, dim=2, eps=1e-12)    # Because buggy mistuned value in Pytorch3d, must re-normalize
        norms = torch.sum(obj_normals * obj_normals, dim=2)  # Dot product
        obj_normals[norms < 0.8] = 0.6   # TODO CRAZY hacky get-around when normal finding fails completely
        self.obj_normals = obj_normals.detach().squeeze().numpy()

        obj_sampled_verts = self.obj_verts[obj_sampled_idx, :]
        obj_sampled_normals = obj_normals[0, obj_sampled_idx, :].detach().numpy()
        hand_normals = hand_mesh.verts_normals_padded()[0, :, :].detach().numpy()

        hand_centroid = np.mean(self.hand_verts, axis=0)
        obj_centroid = np.mean(self.obj_verts, axis=0)

        # Hand features
        hand_one_hot = np.ones((self.hand_verts.shape[0], 1))
        hand_vec_to_closest = hand_nearest.squeeze().numpy() - self.hand_verts
        hand_dist_to_closest = np.expand_dims(np.linalg.norm(hand_vec_to_closest, 2, 1), axis=1)
        hand_dist_along_normal = np.expand_dims(np.sum(hand_vec_to_closest * hand_normals, axis=1), axis=1)
        hand_dist_to_joint = np.expand_dims(self.hand_verts, axis=1) - np.expand_dims(self.hand_joints, axis=0)   # Expand for broadcasting
        hand_dist_to_joint = np.linalg.norm(hand_dist_to_joint, 2, 2)
        hand_dot_to_centroid = np.expand_dims(np.sum((self.hand_verts - obj_centroid) * hand_normals, axis=1), axis=1)

        # Object features
        obj_one_hot = np.zeros((obj_sampled_verts.shape[0], 1))
        obj_vec_to_closest = obj_nearest.squeeze().numpy() - obj_sampled_verts
        obj_dist_to_closest = np.expand_dims(np.linalg.norm(obj_vec_to_closest, 2, 1), axis=1)
        obj_dist_along_normal = np.expand_dims(np.sum(obj_vec_to_closest * obj_sampled_normals, axis=1), axis=1)
        obj_dist_to_joint = np.expand_dims(obj_sampled_verts, axis=1) - np.expand_dims(self.hand_joints, axis=0)   # Expand for broadcasting
        obj_dist_to_joint = np.linalg.norm(obj_dist_to_joint, 2, 2)
        obj_dot_to_centroid = np.expand_dims(np.sum((obj_sampled_verts - hand_centroid) * obj_sampled_normals, axis=1), axis=1)

        # hand_feats = np.concatenate((hand_one_hot, hand_normals, hand_vec_to_closest, hand_dist_to_closest, hand_dist_along_normal, hand_dist_to_joint), axis=1)
        # obj_feats = np.concatenate((obj_one_hot, obj_sampled_normals, obj_vec_to_closest, obj_dist_to_closest, obj_dist_along_normal, obj_dist_to_joint), axis=1)
        hand_feats = np.concatenate((hand_one_hot, hand_dot_to_centroid, hand_dist_to_closest, hand_dist_along_normal, hand_dist_to_joint), axis=1)
        obj_feats = np.concatenate((obj_one_hot, obj_dot_to_centroid, obj_dist_to_closest, obj_dist_along_normal, obj_dist_to_joint), axis=1)

        return hand_feats, obj_feats

    def get_o3d_meshes(self, hand_contact=False, normalize_pos=False):
        """Returns Open3D meshes for visualization
        Draw with: o3dv.draw_geometries([hand_mesh, obj_mesh])"""

        hand_color = np.asarray([224.0, 172.0, 105.0]) / 255
        obj_color = np.asarray([100.0, 100.0, 100.0]) / 255

        obj_centroid = self.obj_verts.mean(0)
        if not normalize_pos:
            obj_centroid *= 0

        hand_mesh = o3dg.TriangleMesh()
        hand_mesh.vertices = o3du.Vector3dVector(self.hand_verts - obj_centroid)
        hand_mesh.triangles = o3du.Vector3iVector(HandObject.closed_faces)
        hand_mesh.compute_vertex_normals()

        if hand_contact and self.hand_contact.mean() != 0:
            util.mesh_set_color(self.hand_contact, hand_mesh)
        else:
            hand_mesh.paint_uniform_color(hand_color)

        obj_mesh = o3dg.TriangleMesh()
        obj_mesh.vertices = o3du.Vector3dVector(self.obj_verts - obj_centroid)
        obj_mesh.triangles = o3du.Vector3iVector(self.obj_faces)
        obj_mesh.compute_vertex_normals()

        if self.obj_contact.mean() != 0:
            util.mesh_set_color(self.obj_contact, obj_mesh)
        else:
            obj_mesh.paint_uniform_color(obj_color)

        return hand_mesh, obj_mesh

    def vis_hand_object(self):
        """Runs Open3D visualizer window"""

        hand_mesh, obj_mesh = self.get_o3d_meshes(hand_contact=True)
        o3dv.draw_geometries([hand_mesh, obj_mesh])
