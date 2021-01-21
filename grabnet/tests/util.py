import os
from os import path as osp
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import pytorch3d
from manopth import manolayer
import open3d
from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion
from open3d import io as o3dio
from open3d import geometry as o3dg
from open3d import utility as o3du
from open3d import visualization as o3dv
from manopth.manolayer import ManoLayer
import trimesh
from grabnet.tools.utils import aa2rotmat, rotmat2aa
import grabnet.tests.util as util


def convert_pca15_aa45(ho, mano_model_in, mano_model_out):
    # Helper function to convert the input format. Input format:
    # mTc is [4,4] rigid transform matrix that encodes large rotations and translation
    # pose is [18] that encodes rotation in first 3 and PCA 15 with curved hand mean
    # Output format:
    # Total rotation in axis angle [3]
    # Translation
    # Fully parameterized [45] axis angle representation

    out_dict = {}

    with torch.no_grad():
        target_verts = torch.Tensor(ho.hand_verts).unsqueeze(0)
        betas = torch.Tensor(ho.hand_beta).unsqueeze(0)

        rot_mTc = torch.Tensor(ho.hand_mTc).unsqueeze(0)[:, :3, :3]
        rot_pose_aa = torch.Tensor(ho.hand_pose[:3]).unsqueeze(0).unsqueeze(0).unsqueeze(0)     # Somehow this wants a Bx1x1x3 input
        rot_pose = aa2rotmat(rot_pose_aa).squeeze(1).squeeze(1).view(-1, 3, 3)

        rot_combined = torch.bmm(rot_mTc, rot_pose)
        rot_combined_aa = rotmat2aa(rot_combined).squeeze(1).squeeze(1)

        hand_pose_in = torch.Tensor(ho.hand_pose[3:]).unsqueeze(0)  # Get 15-dim pca
        mano_out = mano_model_in(global_orient=rot_combined_aa, hand_pose=hand_pose_in, betas=betas, return_full_pose=True)
        hand_pose_out = mano_out.full_pose[:, 3:]     # Get 45-dim full axang representation
        approx_trans = target_verts[:, 0, :] - mano_out.vertices[:, 0, :]

    out_dict['transl'] = approx_trans
    out_dict['global_orient'] = rot_combined_aa
    out_dict['hand_pose'] = hand_pose_out
    out_dict['betas'] = betas

    # verts_rh_gen_cnet = mano_model_out(**out_dict).vertices
    # print('Hand fitting err', np.linalg.norm(verts_rh_gen_cnet.squeeze().detach().numpy() - ho.hand_verts, 2, 1).mean())

    return out_dict


def opt_hand(mano_model, target_verts, hand_pose, hand_trans, hand_rot, betas=None):
    trans_weight = 1
    # Do optimization
    hand_pose = torch.Tensor(hand_pose).clone().detach()
    hand_trans = torch.Tensor(hand_trans).clone().detach() / trans_weight
    hand_rot = torch.Tensor(hand_rot).clone().detach()

    hand_pose.requires_grad = True
    # hand_trans.requires_grad = True
    hand_rot.requires_grad = True

    optimizer = torch.optim.Adam([hand_pose, hand_trans, hand_rot], lr=0.1, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    loss_criterion = torch.nn.L1Loss()

    n_iter = 600
    for it in range(n_iter):
        optimizer.zero_grad()
        out_verts = mano_model(global_orient=hand_rot, hand_pose=hand_pose, transl=hand_trans * trans_weight, betas=betas).vertices
        loss = loss_criterion(out_verts, target_verts)

        if it % 50 == 0:
            vis_pointcloud(out_verts, target_verts)
            print('Opt loss', loss.detach())
        print('Hand trans', hand_trans.detach())

        loss.backward()
        optimizer.step()
        scheduler.step()
        # print(loss.item())
    print('Final loss', loss.detach().data)
    # print('Final trans', hand_trans)

    return hand_pose.detach(), hand_trans.detach() * trans_weight, hand_rot.detach()


def verts_to_name(num_verts):
    num_verts_dict = {100597: 'mouse', 29537: 'binoculars', 100150: 'bowl', 120611: 'camera', 64874: 'cell_phone',
                      177582: 'cup', 22316: 'eyeglasses', 46334: 'flashlight', 35949: 'hammer', 93324: 'headphones',
                      19962: 'knife', 169964: 'mug', 57938: 'pan', 95822: 'ps_controller', 57824: 'scissors',
                      144605: 'stapler', 19708: 'toothbrush', 42394: 'toothpaste', 126627: 'utah_teapot', 90926: 'water_bottle',
                      104201: 'wine_glass', 108248: 'door_knob', 71188: 'light_bulb', 42232: 'banana', 93361: 'apple',
                      8300: 'HO_sugar', 8251: 'HO_soap', 16763: 'HO_mug', 10983: 'HO_mustard', 9174: 'HO_drill',
                      8291: 'HO_cheezits', 8342: 'HO_spam', 10710: 'HO_banana', 8628: 'HO_scissors',
                      148245: 'train_exclude'}

    if num_verts in num_verts_dict:
        return num_verts_dict[num_verts]

    return 'DIDNT FIND {}'.format(num_verts)


def upscale_contact(obj_mesh, obj_sampled_idx, contact_obj):
    """
    When we run objects through our network, they always have a fixed number of vertices.
    We need to up/downscale the contact from this to the original number of vertices
    :param obj_mesh: Pytorch3d Meshes object
    :param obj_sampled_idx: (batch, 2048)
    :param contact_obj:
    :return:
    """
    obj_verts = obj_mesh.verts_padded()
    _, closest_idx, _ = pytorch3d.ops.knn_points(obj_verts, batched_index_select(obj_verts, 1, obj_sampled_idx), K=1)
    upscaled = batched_index_select(contact_obj, 1, closest_idx.squeeze(2))
    return upscaled.detach()


def hack_filedesciptor():
    """
    Sometimes needed if reading datasets very quickly? Don't know why
    Fixes RuntimeError: received 0 items of ancdata
    https://github.com/pytorch/pytorch/issues/973
    """
    torch.multiprocessing.set_sharing_strategy('file_system')

    # import resource
    # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def apply_tform(tform, verts):
    """
    Applies a 4x4 rigid transform to a list of points
    :param tform: tensor (batch, 4, 4)
    :param verts: tensor (batch, N, 3)
    :return:
    """
    verts_homo = torch.cat((verts, torch.ones(verts.shape[0], verts.shape[1], 1, device=verts.device)), 2)
    new_verts = torch.bmm(tform, verts_homo.permute(0, 2, 1)).permute(0, 2, 1)
    return new_verts[:, :, :3]


def apply_rot(rot, verts, around_centroid=False):
    """
    Applies a 3x3 rotation matrix to a list of points
    :param rot: tensor (batch, 3, 3)
    :param verts: tensor (batch, N, 3)
    :return:
    """
    if around_centroid:
        centroid = verts.mean(dim=1)
        verts = verts - centroid

    new_verts = torch.bmm(rot, verts.permute(0, 2, 1)).permute(0, 2, 1)

    if around_centroid:
        new_verts = new_verts + centroid

    return new_verts


def translation_to_tform(translation):
    """
    (batch, 3) to (batch, 4, 4)
    """
    tform_out = pytorch3d.ops.eyes(4, translation.shape[0], device=translation.device)
    tform_out[:, :3, 3] = translation
    return tform_out


def sharpen_contact(c, slope=10, thresh=0.6):
    """
    Apply filter to input, makes into a "soft binary"
    """
    out = slope * (c - thresh) + thresh
    return torch.clamp(out, 0.0, 1.0)


def fit_sigmoid(colors, a=0.05):
    idx = colors > 0
    ci = colors[idx]

    x1 = min(ci)  # Find two points
    y1 = a
    x2 = max(ci)
    y2 = 1-a

    lna = np.log((1 - y1) / y1)
    lnb = np.log((1 - y2) / y2)
    k = (lnb - lna) / (x1 - x2)
    mu = (x2*lna - x1*lnb) / (lna - lnb)
    ci = np.exp(k * (ci-mu)) / (1 + np.exp(k * (ci-mu)))  # Apply the sigmoid
    colors[idx] = ci
    return colors


def subdivide_verts(edges, verts):
    """
    Takes a list of edges and vertices, and subdivides each edge and puts a vert in the middle
    :param edges: (batch, E, 2)?
    :param verts: (batch, V, 3)
    :return: new_verts (batch, E+V, 3)
    """
    # edges = mano_mesh[0].edges_packed()
    new_verts = verts[:, edges].mean(dim=2)
    new_verts = torch.cat([verts, new_verts], dim=1)  # (sum(V_n)+sum(E_n), 3)
    return new_verts


def calc_l2_err(a, b, axis=2):
    if torch.is_tensor(a):
        mse = torch.sum(torch.square(a - b), dim=axis)
        l2_err = torch.sqrt(mse)
        return torch.mean(l2_err, 1)
    else:
        mse = np.linalg.norm(a - b, 2, axis=axis)
        return mse.mean()


def batched_index_select(t, dim, inds):
    """
    Helper function to extract batch-varying indicies along array
    :param t: array to select from
    :param dim: dimension to select along
    :param inds: batch-vary indicies
    :return:
    """
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy) # b x e x f
    return out


def mesh_set_color(color, mesh, colormap=plt.cm.inferno):
    """
    Applies colormap to object
    :param color: Tensor or numpy array, (N, 1)
    :param mesh: Open3D TriangleMesh
    :return:
    """
    # vertex_colors = np.tile(color.squeeze(), (3, 1)).T
    # mesh.vertex_colors = o3du.Vector3dVector(vertex_colors)
    # geometry.apply_colormap(mesh, apply_sigmoid=False)

    colors = np.asarray(color).squeeze()
    if len(colors.shape) > 1:
        colors = colors[:, 0]

    colors[colors < 0.1] = 0.1 # TODO hack to make brighter

    colors = colormap(colors)[:, :3]
    colors = o3du.Vector3dVector(colors)
    mesh.vertex_colors = colors


def aggregate_tforms(tforms):
    """Aggregates a list of 4x4 rigid transformation matricies"""
    device = tforms[0].device
    batch_size = tforms[0].shape[0]

    tform_agg = pytorch3d.ops.eyes(4, batch_size, device=device)
    for tform in tforms:
        tform_agg = torch.bmm(tform, tform_agg)  # Aggregate all transforms

    return tform_agg


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def vis_pointcloud(object_points, hand_points, idx=None, show=True):
    if show:
        plt.switch_backend('TkAgg')
    else:
        plt.switch_backend('agg')

    if idx is None:
        idx = int(np.random.randint(0, hand_points.shape[0]))   # Select random sample from batch

    object_points = object_points[idx, :, :].detach().cpu().numpy()
    hand_points = hand_points[idx, :, :].detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2])
    ax.scatter(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2]) #, c=np.arange(hand_points.shape[0]))

    if show:
        axisEqual3D(ax)
        # plt.axis('off')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    return fig


def get_mano_closed_faces():
    """https://github.com/hassony2/handobjectconsist/blob/master/meshreg/models/manoutils.py"""
    mano_layer = manolayer.ManoLayer(
        joint_rot_mode="axisang", use_pca=False, mano_root='.', center_idx=None, flat_hand_mean=True
    )
    close_faces = torch.Tensor(
        [
            [92, 38, 122],
            [234, 92, 122],
            [239, 234, 122],
            [279, 239, 122],
            [215, 279, 122],
            [215, 122, 118],
            [215, 118, 117],
            [215, 117, 119],
            [215, 119, 120],
            [215, 120, 108],
            [215, 108, 79],
            [215, 79, 78],
            [215, 78, 121],
            [214, 215, 121],
        ]
    )
    closed_faces = torch.cat([mano_layer.th_faces, close_faces.long()])
    # Indices of faces added during closing --> should be ignored as they match the wrist
    # part of the hand, which is not an external surface of the human

    # Valid because added closed faces are at the end
    hand_ignore_faces = [1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551]

    return closed_faces.detach().cpu().numpy() #, hand_ignore_faces


# def text_3d(text, pos, direction=None, degree=0.0, font='DejaVu Sans Mono for Powerline.ttf', font_size=16):
def text_3d(text, pos, direction=None, degree=-90.0, density=10, font='/usr/share/fonts/truetype/freefont/FreeMono.ttf', font_size=10):
    """
    Generate a Open3D text point cloud used for visualization.
    https://github.com/intel-isl/Open3D/issues/2
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    # font_obj = ImageFont.truetype(font, font_size)
    font_obj = ImageFont.truetype(font, font_size * density)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = open3d.geometry.PointCloud()
    pcd.colors = open3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    # pcd.points = o3d.utility.Vector3dVector(indices / 100.0)
    pcd.points = open3d.utility.Vector3dVector(indices / 1000 / density)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


def to_cpu_numpy(obj):
    """Convert torch cuda tensors to cpu, numpy tensors"""
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = to_cpu_numpy(v)
            return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(to_cpu_numpy(v))
        return res
    else:
        raise TypeError("Invalid type for move_to")


def dict_to_device(data, device):
    """Move dict of tensors to device"""
    out = dict()
    for k in data.keys():
        out[k] = data[k].to(device)
    return out


def calc_pck(errors, max_err):
    errs = np.array(errors)
    x = np.linspace(0, max_err, 500)
    y = np.zeros_like(x)

    for i in range(x.shape[0]):
        y[i] = np.sum(errs < x[i]) / errs.shape[0]

    return x, y


def show_histogram(joints_err, verts_err, n_bins=60):
    fig, axs = plt.subplots(2)
    axs[0].set_title('Joints error')
    axs[0].set_ylabel('Frequency')
    axs[0].set_xlabel('Error (mm)')
    axs[0].hist(joints_err * 1000, n_bins)

    axs[1].set_title('Verts error')
    axs[1].set_ylabel('Frequency')
    axs[1].set_xlabel('Error (mm)')
    axs[1].hist(verts_err * 1000, n_bins)

    plt.show()


def show_pck(joints_err, verts_err):
    fig, axs = plt.subplots(2)
    axs[0].set_title('Joints error')
    axs[0].set_ylabel('Percentage of Keypoints')
    axs[0].set_xlabel('Error (mm)')
    x, y = calc_pck(joints_err * 1000, 40)
    axs[0].plot(x, y)

    axs[1].set_title('Verts error')
    axs[1].set_ylabel('Frequency')
    axs[1].set_xlabel('Error (mm)')
    x, y = calc_pck(verts_err * 1000, 40)
    axs[1].plot(x, y)

    plt.show()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)