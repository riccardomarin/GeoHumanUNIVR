## For this assignement you can reuse the usal environment.
## The only new libraries to install should be: tqdm, robust_laplacian

import smplx
import torch 
import open3d as o3d
import numpy as np 
from pytorch3d.loss import (
    chamfer_distance
)
import pickle as pkl
from os import path as osp
from matplotlib import pyplot as plt
from pytorch3d.ops import knn_points
from scipy.sparse.linalg import eigs,eigsh
import tqdm 

# Utility Functions
o3d_float   = o3d.utility.Vector3dVector
o3d_integer = o3d.utility.Vector3iVector
visualizer  = o3d.visualization.draw
TriMesh     = o3d.geometry.TriangleMesh
PointCloud  = o3d.geometry.PointCloud
o3d_read    = o3d.io.read_triangle_mesh
o3d_write   = o3d.io.write_triangle_mesh

# Setting the device
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
# else:
device = torch.device("cpu")
print("WARNING: CPU only, this will be slow!")


target_shape_path = './dataset/registration/target.ply'

######################################################################
##### TASK 1 - v2v registration
##### The target shape has the same vertices of SMPL. This means that the two are in
##### correspondence by their triangulation\parametrization. Tough, we do not know which
##### SMPL parameters generated the target. We can recover them by optimization.

# Creating SMPL layer
smpl_layer_opt = smplx.create('./dataset/body_models/', model_type="smpl", gender='neutral')

# Loading the target shape
target = o3d_read(target_shape_path)

target.compute_vertex_normals()
visualizer([target])

if np.asarray(target.vertices).shape[0] != 6890:
    pass
else:
    # Parsing the target vertices into pytorch
    target_v = torch.tensor(np.asarray(target.vertices), dtype=torch.float32)

    # Initializing optimizer
    optim = torch.optim.Adam(smpl_layer_opt.parameters(), lr=0.05)

    # Setting the number of iterations
    n_iters = 2000
    pbar = tqdm.tqdm(range(0, n_iters))

    # Optimization cycle
    for i in pbar:
        # Getting the SMPL vertices
        predictions = smpl_layer_opt()['vertices']

        # Computing the v2v loss
        loss = torch.sum((predictions - target_v)**2)

        # Backprop
        loss.backward()
        optim.step()
        optim.zero_grad()

        # Logging
        pbar.set_description("Loss: %f" % loss)

    # Storing the result
    v = predictions.detach().cpu().numpy().squeeze()
    f = smpl_layer_opt.faces
    registration = TriMesh(o3d_float(v),o3d_integer(f))
    registration.compute_vertex_normals()
    registration.vertex_colors = o3d_float(v)

    o3d_write('./dataset/output/registration/out_gt.ply', registration)

    # Visualizing the result 
    visualizer([registration, target])

## QUESTIONS
# 1) Do you think we reach a global minimum, or is it possible to obtain even a better alignement?
# 2) Do the intermidiate shapes all represent realistic humans?
######################################################################

##### TASK 2 - Chamfer registration
##### Of course, we generally cannot assume to have a 1:1 correspondence between the shapes
##### hence, a v2v loss cannot be applied. Instead, we can rely on Chamfer distance, as seen
##### during our lectures.

# Creating a SMPL layer
smpl_layer_opt = smplx.create('./dataset/body_models/', model_type="smpl", gender='neutral')

# Loading the target shape
target = o3d_read(target_shape_path)
target.compute_vertex_normals()

# Parsing the target vertices into pytorch
target_v = torch.tensor(np.asarray(target.vertices), dtype=torch.float32)

# Setting the number of iterations
n_iters = 5000
pbar = tqdm.tqdm(range(0, n_iters))

# Initializing optimizer
optim = torch.optim.Adam(smpl_layer_opt.parameters(), lr=0.001)

for i in pbar:
  # Getting the SMPL vertices
  predictions = smpl_layer_opt()['vertices']

  # Define chamfer distance as a loss. 
  # NOTE: Since chamfer can be quite computational expensive, you can subsample the shape
  # by considering one point every 10
  loss, _ = chamfer_distance(predictions[:,::5,:], target_v[::5,:].unsqueeze(0))

  loss.backward()
  optim.step()
  optim.zero_grad()

  # Logging
  pbar.set_description("Loss: %f" % loss)

# Storing the result
v = predictions.detach().cpu().numpy().squeeze()
f = smpl_layer_opt.faces
registration = TriMesh(o3d_float(v),o3d_integer(f))
registration.compute_vertex_normals()
registration.vertex_colors = o3d_float(v)
o3d_write('./dataset/output/registration/out_chamfer.ply', registration)

# Compute an error w.r.t. the GT (v2v distance)
if np.asarray(target.vertices).shape[0] != 6890:
    pass
else:
    GT_error = torch.sum((predictions - target_v)**2)

# Visualizing the result 
visualizer([registration, target])

## QUESTIONS
# 1) Compared to the v2v fitting, we are using more iterations and a lower learning rate. Why?
# 2) Do you think that using some of the regularizations that we have seen during the lecture (e.g., edge loss, laplacian loss, normal loss,...) can help? 
#    If yes, why and which? If no, why?

######################################################################

##### TASK 3 - Chamfer registration + Pose Prior
##### To regularize the optimization, we can rely on a learned prior on the SMPL parameters. 
##### Specifically, someone has fitted a Gaussian model to the distribution of the SMPL poses (\theta parameters)
##### and stored the mean and the inverse of the covariance matrix into './dataset/priors/body_prior.pkl'.
##### During the optimization, we can use the (squared) Mahalanobis distance to assess how plausible is a certain pose.
##### https://en.wikipedia.org/wiki/Mahalanobis_distance
##### 

# Create SMPL layer
smpl_layer_opt = smplx.create('./dataset/body_models/', model_type="smpl", gender='neutral')

# Loading the target shape
target = o3d_read(target_shape_path)
target.compute_vertex_normals()

# Parsing the target vertices into pytorch
target_v = torch.tensor(np.asarray(target.vertices), dtype=torch.float32)

# Loading the pose prior
body_prior = pkl.load(open(osp.join('dataset/prior', 'body_prior.pkl'), 'rb'), encoding='latin')

# Loading the mean of a gaussian distribution
gmm_mean = torch.from_numpy(body_prior['mean']).float().unsqueeze(0).to(device)

# Loading the inverse of covariance matrix (Σ⁻¹). REMARK: it is stored as Cholesky decomposition
gmm_precision_ch = torch.from_numpy(body_prior['precision']).float().to(device)
gmm_precision = gmm_precision_ch @ gmm_precision_ch.T

# Setting the number of iterations
n_iters = 5000
pbar = tqdm.tqdm(range(0, n_iters))

# Initializing optimizer
optim = torch.optim.Adam(smpl_layer_opt.parameters(), lr=0.001)

# Defyining the squared mahalanobis distance
def mahalanobis(u, mu, cov):
    delta = u - mu
    m = torch.matmul(delta.squeeze(), torch.matmul(cov, delta.squeeze()))
    return m

# Running optimization with annealing on prior regularization
# epochs from 0 to 1500:    0.01
# epochs from 1500 to 3500: 0.001
# epochs from 3500 to 4500: 0.0001
# After 4500:               0.0

for i in pbar:
  # Getting the SMPL vertices
  predictions  = smpl_layer_opt()['vertices']

  # Getting the SMPL pose
  pose         = smpl_layer_opt.body_pose 

  # Computing the prior loss
  loss_prior   = mahalanobis(pose[:, :63],gmm_mean, gmm_precision) 

  # Using chamfer distance as a loss. 
  # NOTE: Since chamfer can be quite computational expensive, you can subsample the shape
  # by considering one point every 10
  loss_data, _ = chamfer_distance(predictions[:,::5,:], target_v[::5,:].unsqueeze(0))

  # Annealing
  if i < 1500:
    wh = 0.01
  if i > 1500:
    wh = 0.001
  if i > 3500:
    wh = 0.0001
  if i > 4500:
    wh = 0

  # Loss composition
  loss = loss_data + loss_prior * wh 

  # Backward pass
  loss.backward()
  optim.step()
  optim.zero_grad()

  # Logging
  pbar.set_description(f"Cham_Loss: {loss_data:.4f} / prior_Loss: {loss_prior:.4f}")

# Storing the result
v = predictions.detach().cpu().numpy().squeeze()
f = smpl_layer_opt.faces
registration = TriMesh(o3d_float(v),o3d_integer(f))
registration.compute_vertex_normals()
registration.vertex_colors = o3d_float(v)
o3d_write('./dataset/output/registration/out_chamfer+pose.ply', registration)

# Compute an error w.r.t. the GT (v2v distance)
if np.asarray(target.vertices).shape[0] != 6890:
    pass
else:
    GT_error = torch.sum((predictions - target_v)**2)

# Visualizing the result
visualizer([registration, target])

## QUESTIONS
# 1) Run it again but keep the regularization weight constant across the optimization. What happen?
# 2) How the GT_error and the Chamfer loss changed from Task 2?
# 3) What do you think is the problem that limit this optimization?

