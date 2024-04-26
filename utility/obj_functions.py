import numpy as np

def loadobj(path):
    vertices = []
    triangles = []

    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue  # Skip comments
                pieces = line.split()
                if not pieces:
                    continue  # Skip empty lines
                if pieces[0] == 'v' and len(pieces) >= 4:
                    vertices.append([float(x) for x in pieces[1:4]])  # Extract vertices
                elif pieces[0] == 'f' and len(pieces) >= 4:
                    triangles.append([int(x.split('/')[0]) - 1 for x in pieces[1:]])  # Extract triangles
                elif pieces[0] == 'vn':
                    pass  # Ignore vertex normals for now
                else:
                    print("Warning: Unrecognized line in obj file:", line.strip())  # Warn about unrecognized lines
    except Exception as e:
        print("Error loading obj file:", e)  # Handle file loading errors

    return np.array(vertices, dtype=np.float32), np.array(triangles, dtype=np.int32)  # Return vertices and triangles


def writeobj(filepath, vertices, triangles):
    with open(filepath, "w") as f:
        for vertex in vertices:
            f.write("v {} {} {}\n".format(*vertex))  # Write vertices to file
        for triangle in triangles:
            f.write("f {} {} {}\n".format(triangle[0] + 1, triangle[1] + 1, triangle[2] + 1))  # Write triangles to file

def compute_normal(vertices, indices):
    eps = 1e-10
    vn = np.zeros(vertices.shape, dtype=np.float32)

    for index in indices:
        v0, v1, v2 = vertices[index]
        e1 = v1 - v0
        e2 = v2 - v0
        e1_len = np.linalg.norm(e1)
        e2_len = np.linalg.norm(e2)
        side_a = e1 / (e1_len + eps)
        side_b = e2 / (e2_len + eps)
        fn = np.cross(side_a, side_b)
        fn = fn / (np.linalg.norm(fn) + eps)

        angle = np.where(np.sum(side_a * side_b) < 0,
                         np.pi - 2.0 * np.arcsin(0.5 * np.linalg.norm(side_a + side_b)),
                         2.0 * np.arcsin(0.5 * np.linalg.norm(side_b - side_a)))
        sin_angle = np.sin(angle)
        contrib = fn * sin_angle / ((e1_len * e2_len) + eps)
        
        for i, idx in enumerate(index):
            vn[idx] += contrib[i]

    vn = vn / (np.linalg.norm(vn, axis=-1, keepdims=True) + eps)
    return None, vn  # Only return vertices_normals

