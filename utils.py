from re import L
import numpy as np
import open3d
import os
import camera
import pickle as pkl

from igibson.simulator import load_without_pybullet_vis
osp = os.path

test_objects = ['mug', 'pan', 'wine_glass']


base_dir = osp.expanduser(osp.join('~', 'deepgrasp_data'))
use_data_dirs = \
    [osp.join(base_dir, 'data')]*28 + \
    [osp.join(base_dir, 'data3')] + \
    [osp.join(base_dir, 'data2')]*4 + \
    [osp.join(base_dir, 'data3')]*17
handoff_data_dirs = \
    [osp.join(base_dir, 'data')]*24 + \
    [osp.join(base_dir, 'data3')]*5 +\
    [osp.join(base_dir, 'data2')]*4 + \
    [osp.join(base_dir, 'data3')]*17

def camera_to_world(view_matrix, cam_coords):
  pose = camera.cam_view2pose(view_matrix)
  world_coords = cam_coords @ pose[:3,:3].T + pose[:3,3]
  return world_coords


def texture_proc(colors, a=0.05, invert=False):
  idx = colors > 0
  ci = colors[idx]
  if len(ci) == 0:
    return colors
  if invert:
    ci = 1 - ci
  # fit a sigmoid
  x1 = min(ci); y1 = a
  x2 = max(ci); y2 = 1-a
  lna = np.log((1 - y1) / y1)
  lnb = np.log((1 - y2) / y2)
  k = (lnb - lna) / (x1 - x2)
  mu = (x2*lna - x1*lnb) / (lna - lnb)
  # apply the sigmoid
  ci = np.exp(k * (ci-mu)) / (1 + np.exp(k * (ci-mu)))
  colors[idx] = ci
  return colors

def degenerate_masks(preds):
  filter=np.zeros(preds.shape[0])

  for i, pred in enumerate(preds):
    if np.any(pred!=0):
      filter[i]=1
  
  preds = preds[filter.astype(bool)]

  print(preds.shape)

  return preds 

def discretize_texture(c, thresh=0.4, have_dontcare=True):
  idx = c > 0
  if sum(idx) == 0:
    return c
  ci = c[idx]
  c[:] = 2 if have_dontcare else 0
  ci = ci > thresh
  c[idx] = ci
  return c

def load_from_pkl(data):
  """
    Load geoms and predictions from pickle. 

    :param data: Data returned from pick.load()
  """ 
  geom = data['geom']
  preds = data['tex_preds']

  return geom, preds

def show_voxel_texture(geom, tex_preds):
  cmap = np.asarray([[0, 0, 1], [1, 0, 0]])
  z, y, x = np.nonzero(geom[0])
  pts = np.vstack((x, y, z)).T
  for tex_pred in tex_preds:
    tex_pred = np.argmax(tex_pred, axis=0)
    tex_pred = tex_pred[z, y, x]
    tex_pred = cmap[tex_pred]
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(pts)
    pc.colors = open3d.utility.Vector3dVector(tex_pred)
    open3d.visualization.draw_geometries([pc])

def show_pointcloud_from_pkl(file):

  filename = osp.expanduser(file)

  with open(filename, 'rb') as f:
    d = pkl.load(f)  
  geom, preds = load_from_pkl(d)
  show_voxel_texture(geom, preds)

def show_pointcloud(pts, colors,
    cmap=np.asarray([[0,0,1],[1,0,0],[0,0,1]])):
  colors = np.asarray(colors)
  if (colors.dtype == int) and (colors.ndim == 1) and (cmap is not None):
    colors = cmap[colors]
  if colors.ndim == 1:
    colors = np.tile(colors, (3, 1)).T

  pc = open3d.geometry.PointCloud()
  pc.points = open3d.utility.Vector3dVector(np.asarray(pts))
  pc.colors = open3d.utility.Vector3dVector(colors)

  open3d.visualization.draw_geometries([pc])
