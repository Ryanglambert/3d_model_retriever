import numpy as np


def check_header(a):
    "check header of *.off file"
    header = a[0]
    assert header.strip('\n') == 'OFF'


def parse_vertices_faces_edges(off_list):
    "Gets num vertices, faces, edges from lines of *.off file"
    first = off_list[1]
    vertices, faces, edges = [int(i) for i in first.split()]
    return vertices, faces, edges


def load_off_file(path):
    with open(path, 'r') as f:
        return f.readlines()


def parse_coordinates(a):
    "Returns vertex coordinate data"
    vert_idx, _, _ = parse_vertices_faces_edges(a)
    vert_start = 2 # coordinates always start at this index
    vert_end = vert_start + vert_idx
    coord_strings = a[vert_start: vert_end]
    new_coords = []
    for coord_string in coord_strings:
        new_coords.append([float(i) for i in coord_string.split()])
    return np.array(new_coords)


def parse_face_indeces(a):
    "Returns face indeces np.array([num_sides, coord_index1, ..., coord_index_num_sides])"
    vert_idx, faces_idx, _ = parse_vertices_faces_edges(a)
    vert_start = 2 # coordinates always start at this index
    face_start = vert_start + vert_idx
    face_end = vert_start + vert_idx + faces_idx
    face_strings = a[face_start:face_end]
    new_indeces = []
    for face_string in face_strings:
        new_indeces.append([int(i) for i in face_string.split()])
    return np.array(new_indeces)
