import torch
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt

data = tio.ScalarImage('data/00-P.mhd')
transform = tio.transforms.Resample(1)
data = transform(data)
transform = tio.transforms.RescaleIntensity((0, 1))
data = transform(data)
data

#creating a random vector 
needle_vec = np.array([0, 0.5, 0]) #needle as a vector
standard_i = np.array([1, 0, 0])
needle_vec /= np.linalg.norm(needle_vec)
norm_vec = np.cross(needle_vec, standard_i)
norm_vec /= np.linalg.norm(norm_vec)
print(norm_vec)
point = np.array([0, 0.2, 0.25]) #position of the needle


# plane coordinate in needle frame
render_image_W, render_image_H = 512, 512
X, Y = np.meshgrid(np.arange(render_image_W), np.arange(render_image_H))
Y -= render_image_W // 2
Z = np.zeros_like(X)
#TODO
Z_slant = point[2] - (norm_vec[0] * (X - point[0]) + norm_vec[1] * (Y - point[1])) / norm_vec[2]
X, Y, Z, Z_slant= X.flatten(), Y.flatten(), Z.flatten(), Z_slant.flatten()
plane_points = np.stack([X, Y, Z_slant], axis=1) / 1000

# transform to volume frame
y_vec = np.cross(norm_vec, [1, 0 ,0])
y_vec /= np.linalg.norm(y_vec)
V_T_N = np.array([
    [1, 0, 0, point[0]],  # X-axis (fixed)
    [0, 1, 0, point[1]],  # Y-axis (up direction)
    [0, 0, 1, point[2]],  # Z-axis (plane normal)
    [0, 0, 0, 1]   # Homogeneous row
])

plane_points = np.concatenate([plane_points, np.ones((plane_points.shape[0], 1))], axis=1)
plane_points_in_volume = np.einsum('ij,kj->ki', V_T_N, plane_points)[:,:3]

# sample value using nearest grid point
volume_aabb = np.array([[0, 0, 0], data.shape[1:]])/1000
plane_points_inside_volume_mask = np.logical_and(np.all(plane_points_in_volume > volume_aabb[0], axis=1), np.all(plane_points_in_volume < volume_aabb[1], axis=1))
sample_indices = np.floor((plane_points_in_volume[plane_points_inside_volume_mask] * 1000))
sample_values = data.tensor[0, sample_indices[:, 0], sample_indices[:, 1], sample_indices[:, 2]]

plane_pixel_values = np.zeros((plane_points_in_volume.shape[0]))
plane_pixel_values[plane_points_inside_volume_mask] = sample_values
plane_pixel_values = plane_pixel_values.reshape(render_image_W, render_image_H)

plt.imshow(plane_pixel_values)
plt.show()


