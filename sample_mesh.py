#!/usr/bin/env python

import trimesh
import numpy as np
import torch
from pytorch3d.ops import knn_points
import time
import os
import pickle
import sys


def compute_mean_curvature(data: torch.Tensor, k: int = 20) -> torch.Tensor:
    coords = data[:, :3]
    normals = torch.nn.functional.normalize(data[:, 3:], dim=1)
    coords_batch = coords.unsqueeze(0)
    knn_result = knn_points(coords_batch, coords_batch, K=k + 1, return_nn=True)
    knn_neighbors = knn_result.knn[0][:, 1:, :]
    coords_expanded = coords.unsqueeze(1)
    neighbor_diffs = knn_neighbors - coords_expanded
    laplacian = neighbor_diffs.mean(dim=1)
    curvature = 0.5 * torch.abs(torch.sum(laplacian * normals, dim=1))
    return curvature


def compute_mean_curvature_divergence(data: torch.Tensor, k: int = 20) -> torch.Tensor:
    coords = data[:, :3]
    normals = torch.nn.functional.normalize(data[:, 3:], dim=1)
    coords_batch = coords.unsqueeze(0)
    knn_result = knn_points(coords_batch, coords_batch, K=k + 1, return_nn=False)
    knn_indices = knn_result.idx[0][:, 1:]
    neighbor_coords = coords[knn_indices]
    neighbor_normals = normals[knn_indices]
    coords_expanded = coords.unsqueeze(1)
    normals_expanded = normals.unsqueeze(1)
    diff_coords = neighbor_coords - coords_expanded
    diff_normals = neighbor_normals - normals_expanded
    sq_dists = torch.sum(diff_coords ** 2, dim=2) + 1e-8
    dot_prod = torch.sum(diff_normals * diff_coords, dim=2)
    quotients = dot_prod / sq_dists
    divergence = quotients.mean(dim=1)
    curvature = 0.5 * divergence.abs()
    return curvature


def compute_curvature_gradient(data: torch.Tensor, curvature: torch.Tensor, k: int = 20) -> torch.Tensor:
    coords = data[:, :3]
    coords_batch = coords.unsqueeze(0)
    knn_result = knn_points(coords_batch, coords_batch, K=k + 1, return_nn=False)
    knn_indices = knn_result.idx[0][:, 1:]
    neighbor_coords = coords[knn_indices]
    neighbor_curvatures = curvature[knn_indices]
    coords_expanded = coords.unsqueeze(1)
    curvature_expanded = curvature.unsqueeze(1)
    delta_coords = neighbor_coords - coords_expanded
    delta_curvatures = neighbor_curvatures - curvature_expanded
    sq_dists = torch.sum(delta_coords ** 2, dim=2) + 1e-8
    weighted_diff = (delta_curvatures.unsqueeze(2) * delta_coords) / sq_dists.unsqueeze(2)
    grad_vector = weighted_diff.mean(dim=1)
    grad_norm = torch.norm(grad_vector, dim=1)
    avg_delta = delta_curvatures.mean(dim=1)
    sign_factor = torch.sign(avg_delta)
    signed_grad = sign_factor * grad_norm
    return signed_grad


def get_adjacent_faces(mesh, edge_index):
    edge = mesh.edges_sorted[edge_index]
    face_adj_index = mesh.face_adjacency_edges_tree.query([edge])[1]
    adjacent_face_indices = mesh.face_adjacency[face_adj_index]
    return adjacent_face_indices.squeeze()


def sample_high_curvature_points(mesh):
    curvature_threshold = np.deg2rad(2.5)
    high_curvature_data = []
    for edge_index in range(len(mesh.edges_sorted)):
        adjacent_faces = get_adjacent_faces(mesh, edge_index)
        if len(adjacent_faces) < 2:
            print(edge_index)
            continue
        face1_index, face2_index = adjacent_faces
        normal1 = mesh.face_normals[face1_index]
        normal2 = mesh.face_normals[face2_index]
        edge = mesh.edges_sorted[edge_index]
        start = mesh.vertices[edge[0]]
        end = mesh.vertices[edge[1]]
        angle = np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))
        if angle > curvature_threshold:
            high_curvature_data.append((edge_index, start, end, face1_index, face2_index, normal1, normal2))
        else:
            if angle < 0:
                print(angle)

    print("Number of high-curvature edges found:", len(high_curvature_data))
    edge_points = []

    for data in high_curvature_data:
        edge_index, start, end, face1_index, face2_index, normal1, normal2 = data
        segment_vec = end - start
        seg_length = np.linalg.norm(segment_vec)
        if seg_length == 0:
            continue
        distances = np.arange(0, seg_length, 0.001)
        if distances.size == 0 or distances[0] != 0:
            distances = np.insert(distances, 0, 0)
        if distances[-1] != seg_length:
            distances = np.append(distances, seg_length)
        t_values = distances / seg_length

        for t in t_values:
            point = start + t * segment_vec
            avg_normal = normal1 + normal2
            norm_avg = np.linalg.norm(avg_normal)
            avg_normal = avg_normal / norm_avg if norm_avg > 0 else np.array([0, 0, 0])
            edge_points.append(np.hstack((point, avg_normal, [edge_index], [-1])))

    edge_points = np.vstack(edge_points)
    print(f'Num edge points: {edge_points.shape[0]}')
    return edge_points


def symmetric_log_transform(x):
    return np.sign(x) * np.log1p(np.abs(x))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    errors = []
    for file in os.listdir(input_dir):
        output_file = os.path.join(output_dir, file.replace('.stl', '_processed.npy'))
        label_file = os.path.join("path/to/labeled_meshes", file.replace('.stl', '_labeled.npy'))
        if not os.path.exists(label_file) and file.endswith('.stl'):
            try:
                print(file)
                input_file = os.path.join(input_dir, file)
                mesh = trimesh.load_mesh(input_file)
                print("Mesh area:", mesh.area)
                edge_pts = sample_high_curvature_points(mesh)
                surface_sampled_points, face_indices = trimesh.sample.sample_surface_even(mesh, 150000)
                normals = mesh.face_normals[face_indices]
                surface_pts = np.hstack((surface_sampled_points, normals))
                neg_ones_column = -np.ones((surface_pts.shape[0], 1))
                surface_pts = np.hstack((surface_pts, neg_ones_column, np.expand_dims(face_indices, axis=1)))
                all_points = np.vstack((edge_pts, surface_pts))
                print(f'Total num points: {all_points.shape[0]}')
                data = torch.tensor(all_points[:, :-2], dtype=torch.float32).to('cuda')

                start = time.time()
                H_10 = compute_mean_curvature_divergence(data, 15)
                h10_np = np.expand_dims(symmetric_log_transform(H_10.cpu().numpy()), axis=1)
                H_20 = compute_mean_curvature_divergence(data, 25)
                h20_np = np.expand_dims(symmetric_log_transform(H_20.cpu().numpy()), axis=1)
                print('Curvature time: ', time.time() - start)

                grad_5 = compute_curvature_gradient(data, torch.tensor(h20_np.squeeze(), dtype=torch.float32).cuda(), k=10)
                grad_10 = compute_curvature_gradient(data, torch.tensor(h20_np.squeeze(), dtype=torch.float32).cuda(), k=15)
                grad_20 = compute_curvature_gradient(data, torch.tensor(h20_np.squeeze(), dtype=torch.float32).cuda(), k=25)

                grad_5_np = np.expand_dims(symmetric_log_transform(grad_5.cpu().numpy()), axis=1)
                grad_10_np = np.expand_dims(symmetric_log_transform(grad_10.cpu().numpy()), axis=1)
                grad_20_np = np.expand_dims(symmetric_log_transform(grad_20.cpu().numpy()), axis=1)

                final_data = np.hstack((all_points, h10_np, h20_np, grad_5_np, grad_10_np, grad_20_np))
                np.random.shuffle(final_data)
                print(final_data[0])

                np.save(output_file, final_data)
                print(f'Saved to: {output_file}')
            except Exception as e:
                print(f'ERROR: {str(e)}')
                print(file)
                errors.append(file)
        else:
            print(f'Already processed: {file}')

    error_log = os.path.join(input_dir, 'mesh_pcd_labeling_errors.pkl')
    with open(error_log, 'wb') as f:
        pickle.dump(errors, f)
