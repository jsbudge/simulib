import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplib
import mmap
import pathlib
import re
from glob import glob
from tqdm import tqdm

mplib.use('TkAgg')


def loadGLTF(fnme):
    build_mesh = o3d.io.read_triangle_mesh(fnme)
    fpath = pathlib.Path(fnme)
    with open(f'{fpath.parent}/{fpath.stem}.targ', 'w') as f:
        f.write(fnme + '\n')
        for val in list(set(np.asarray(build_mesh.triangle_material_ids))):
            f.write(f'{val} {val} 1000000. .0017\n')
    return build_mesh


def loadOBJ(fnme):
    with open(fnme, "r+") as f:
        buf = mmap.mmap(f.fileno(), 0)

    lines = 0
    while buf.readline():
        lines += 1

    vertices = []
    vertex_normals = []
    vertex_textures = []
    material_key = {'Default': [0, 1000000., .0017]}
    o = {}
    curr_o = None
    curr_mat = 'Default'
    faces = []
    face_normals = []
    face_material = []
    idx = 1
    mat_idx = 0
    buf.seek(0)
    while idx <= lines:
        line = buf.readline().decode('utf-8').strip()
        if isinstance(line, str):
            if line.startswith('v '):
                try:
                    vertices.append(list(map(float, line.split()[1:4])))
                except ValueError:
                    vertices.append([float(re.findall('-?[0-9]+\.?[0-9]*', l)[0]) for l in line.split()[1:4]])
            elif line.startswith('vn '):
                vertex_normals.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                # Face data (e.g., f v1 v2 v3)
                parts = line.split()
                fces = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                fc_norms = [int(p.split('/')[2]) - 1 if len(p.split('/')) == 3 else 0 for p in parts[1:]]
                if len(fces) > 3:
                    for n in range(len(fces) - 2):
                        faces.append([fces[0], fces[n + 1], fces[n + 2]])
                        face_normals.append([fc_norms[0], fc_norms[n + 1], fc_norms[n + 2]])
                        face_material.append(material_key[curr_mat][0])
                else:
                    faces.append(fces)
                    face_normals.append(fc_norms)
                    face_material.append(material_key[curr_mat][0])
            elif line.startswith('o '):
                # Add the old o to the dict
                if curr_o is not None and len(faces) > 0:
                    o[curr_o] = [np.array(faces), np.array(face_normals), np.array(face_material)]
                # Reset everything for the new o
                faces = []
                face_normals = []
                face_material = []
                curr_o = line[2:].strip()
                if curr_o not in material_key.keys():
                    mat_idx += 1
                    material_key[curr_o] = [mat_idx, 1000000., .0017]
                curr_mat = curr_o
            elif line.startswith('usemtl '):
                parts = line.split(' ')
                if parts[1] not in material_key.keys():
                    mat_idx += 1
                    material_key[parts[1]] = [mat_idx, 1000000., .0017]
                curr_mat = parts[1].strip()
            '''elif line.startswith('vt '):
                vertex_textures.append(list(map(float, line.split()[1:4])))'''

        idx += 1
    if curr_o is None:
        curr_o = 'Default'
    else:
        material_key[curr_o] = [mat_idx, 1000000., .0017]
    o[curr_o] = [np.array(faces), np.array(face_normals), np.array(face_material)]
    vertex_normals = np.array(vertex_normals)
    # Get material numbers correct
    mat_nums = []
    triangles = []
    # tri_norms = []
    for val in o.values():
        mat_nums.append(val[2])
        triangles.append(val[0])
        # tri_norms.append(vertex_normals[val[1]].mean(axis=1))
    # tri_norms = np.concatenate(tri_norms)
    # tri_norms = tri_norms / np.linalg.norm(tri_norms, axis=1)[:, None]
    build_mesh = o3d.geometry.TriangleMesh()
    build_mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    build_mesh.triangles = o3d.utility.Vector3iVector(np.concatenate(triangles).astype(int))
    build_mesh.triangle_material_ids = o3d.utility.IntVector(list(np.concatenate(mat_nums).astype(int)))
    # build_mesh.triangle_normals = o3d.utility.Vector3dVector(tri_norms)
    build_mesh.compute_vertex_normals()

    fpath = pathlib.Path(fnme)
    with open(f'{fpath.parent}/{fpath.stem}.targ', 'w') as f:
        f.write(fnme + '\n')
        for key, val in material_key.items():
            f.write(f'{key} {val[0]} {val[1]} {val[2]}\n')
    return build_mesh, o, np.array(vertices), material_key


# target_meshes = glob('/home/jeff/Documents/target_meshes/*')
target_meshes = ['/home/jeff/Documents/target_meshes/frigate.obj']
for f in tqdm(target_meshes):
    if f.endswith('.obj'):
        check = loadOBJ(f)
    elif f.endswith('.gltf'):
        check = loadGLTF(f)