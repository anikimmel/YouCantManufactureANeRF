#!/usr/bin/env python

import os
import numpy as np
import pickle

from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.Geom import Geom_Plane
from concurrent.futures import TimeoutError
import time
import shutil
import traceback
from occwl.io import load_step
import multiprocessing as mp


def face_is_plane(face):
    surface = face.surface()
    return surface.DynamicType().Name() == Geom_Plane.__name__


def is_point_on_surface(face, point, tolerance=1e-6):
    surface = face.surface()
    analyzer = ShapeAnalysis_Surface(surface)
    uv = analyzer.ValueOfUV(point, tolerance)
    u = uv.X()
    v = uv.Y()
    projected_point = analyzer.Value(u, v)
    distance = point.Distance(projected_point)
    print(f'Distance: {distance}')
    return distance < tolerance, distance


def label_point(args):
    faces, point_coords = args
    faces = shape.faces()
    if (len(list(faces)) == 0):
        print(point_coords)
        print("NO FACES!!!!! **************")
    closest_dist_yet = np.inf
    index_of_closest = -1
    closest_shape = None
    for i, s in enumerate(shape.faces()):
        closest_point_data = s.find_closest_point_data(np.asarray([point_coords[0], point_coords[1], point_coords[2]]))
        if closest_point_data.distance < closest_dist_yet:
            closest_shape = s
            closest_dist_yet = closest_point_data.distance
            index_of_closest = i + 1

    if not closest_shape:
        print("closest_shape is empty")
        return point_coords[0], point_coords[1], point_coords[2], -1
    tolerance = 0.06 if face_is_plane(closest_shape) else 0.08
    if closest_dist_yet < tolerance:
        return point_coords[0], point_coords[1], point_coords[2], index_of_closest
    return point_coords[0], point_coords[1], point_coords[2], -1


def init_worker(faces_arg, file_path_arg, global_timeout_arg, start_time_arg):
    global faces, file_path, global_timeout, start_time
    faces = shape.faces()
    if (len(list(faces)) == 0):
        print("NO FACES!!!!! **************")
    file_path = file_path_arg
    global_timeout = global_timeout_arg
    start_time = start_time_arg


def work(point):
    try:
        if time.time() - start_time >= global_timeout:
            raise TimeoutError("Global timeout exceeded.")
        faces = shape.faces()
        if (len(list(faces)) == 0):
            print("NO FACES!!!!! **************")
        result = label_point((faces, point))
        if result[3] == -1:
            return "mislabeled", result
        return "success", result

    except Exception as e:
        return "error", str(e)


def parallel_labeling(faces, point_coords, file_path, num_workers=64, global_timeout=1200):
    labeled_points = []
    count_mislabeled = 0
    start_time = time.time()

    try:
        with mp.Manager() as manager:
            with mp.Pool(
                    processes=num_workers,
                    initializer=init_worker,
                    initargs=(faces, file_path, global_timeout, start_time),
            ) as pool:
                for status, result in pool.imap_unordered(work, point_coords, chunksize=500):
                    if status == "error":
                        raise Exception(result)

                    if status == "mislabeled":
                        count_mislabeled += 1
                        print(f"count mislabeled: {count_mislabeled}")

                        if count_mislabeled > 100:
                            print('Too many mislabeled points')
                            name = os.path.basename(file_path)
                            pool.terminate()
                            pool.join()
                            shutil.move(file_path, os.path.join("path/to/mislabeled/", name))
                            return labeled_points, count_mislabeled

                    labeled_points.append(result)

                pool.close()
                pool.join()

    except TimeoutError:
        name = os.path.basename(file_path)
        shutil.move(file_path, os.path.join("path/to/timeout_errors/", name))
        print(f"Global timeout of {global_timeout} seconds exceeded.")
        raise

    except Exception as e:
        name = os.path.basename(file_path)
        traceback.print_exc()
        shutil.move(file_path, os.path.join("path/to/mislabeled/", name))
        print(f"An error occurred: {str(e)}")
        raise e

    return labeled_points, count_mislabeled


pcd_dir = "path/to/pcd_dir/"
step_file_path = "path/to/step_files/"
output = "path/to/output_dir/"

if not os.path.exists(output):
    os.mkdir(output)

with open('path/to/test_data.pkl', 'rb') as file:
    test_data = pickle.load(file)

errors = []
for data in test_data:
    try:
        file = data[0].replace('_labeled.npy', '_labeled_cleaned.npy')
        file_path = os.path.join(pcd_dir, file)
        print(f'Processing: {file_path}')
        output_path = os.path.join(output, file.replace('_labeled_cleaned.npy', '_gt_segments.npy'))
        if not os.path.exists(output_path):
            point_coords = np.load(file_path)
            print(point_coords[0])
            labels = point_coords[:, 6].astype(int)
            print(labels)
            print(f'Number of points: {len(point_coords)}')
            step_file_name = os.path.join(step_file_path, file.replace('_labeled_cleaned.npy', '.stp'))
            if not os.path.exists(step_file_name):
                step_file_name = os.path.join(step_file_path, file.replace('_labeled_cleaned.npy', '.step'))

            shape = load_step(step_file_name)[0]
            shape = shape.split_all_closed_faces(num_splits=0)
            shape = shape.scale_to_box(1)
            faces = shape.faces()
            print(f'Number of faces: {shape.num_faces()}')
            nb_points = point_coords[np.where(labels == -1)]
            b_points = point_coords[np.where(labels != -1)]
            print(f'Number of non-boundary points: {len(nb_points)}')
            try:
                start = time.time()
                labeled_points, count_mislabeled = parallel_labeling(shape.faces(), nb_points, file_path)
                end = time.time()
                print(f'Time elapsed for labelling: {end-start}')
                if count_mislabeled < 100:
                    with open(output_path, 'wb') as f:
                        pickle.dump(np.asarray(labeled_points), f)
                    print(f'Saved to: {output_path}')
                else:
                    print("TOO MANY MISLABELED POINTS")
                    errors.append({'file': file, 'error': 'Too many mismatched points'})
            except Exception as e:
                print(str(e))
                errors.append({'file': file, 'error': str(e)})
                continue
        else:
            print("Already Processed")
    except Exception as e:
        print(str(e))
        errors.append({'file': file, 'error': str(e)})
        continue

with open(os.path.join(pcd_dir, 'pcd_labeling_errors.pkl'), 'wb') as f:
    pickle.dump(errors, f)
print('Saved to: pcd_labeling_errors.pkl')
