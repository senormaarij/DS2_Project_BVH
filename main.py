import numpy as np
import sys
import os
from bvh.BVH import *
from utility.obj_functions import *


def create_box(nodelist):
    # Create a box mesh based on AABB of a BVHNode
    botleft = nodelist.aabb.botleft.flatten()[:3]
    topright = nodelist.aabb.topright.flatten()[:3]
    vertex = np.array([botleft, [botleft[0], topright[1], botleft[2]], [topright[0], topright[1], botleft[2]],
                       [topright[0], botleft[1], botleft[2]], [botleft[0], botleft[1], topright[2]],
                       [botleft[0], topright[1], topright[2]], topright, [topright[0], botleft[1], topright[2]]])
    face = np.array([[0, 4, 1], [1, 4, 5], [4, 7, 5], [5, 7, 6], [3, 7, 6], [2, 3, 6], [0, 2, 3], [0, 1, 2],
                     [1, 6, 2], [1, 5, 6], [0, 7, 4], [0, 3, 7]])
    return vertex, face


def save_each_layer(bvhnodelist):
    ## first layer
    layer = 0
    print("layer:{}".format(layer))
    layer_v, layer_f = create_box(bvhnodelist[0])
    writeobj(os.path.join(outpath, "layer_{}.obj".format(layer)), layer_v, layer_f)
    nodes = [0]
    layer += 1

    ## func for getting child node
    def return_child(node, nodes):
        new_nodes = []

        for i in nodes:
            child1 = node[i].child1
            child2 = node[i].child2

            if child1 != -1:
                new_nodes.append(child1)
            if child2 != -1:
                new_nodes.append(child2)
            if child1 == -1 and child2 == -1:
                new_nodes.append(i)

        return new_nodes

    ## create new mesh for each layer
    prev = -1  # Initialize prev outside the loop
    while len(return_child(bvhnodelist, nodes)) != prev:
        print("layer:{}".format(layer))

        # Update node list
        nodes = return_child(bvhnodelist, nodes)
        for i, index in enumerate(nodes):

            temp_v, temp_f = create_box(bvhnodelist[index])
            if i == 0:
                layer_v = temp_v
                layer_f = temp_f
            else:
                layer_v = np.vstack((layer_v, temp_v))
                layer_f = np.vstack((layer_f, temp_f + (8 * i)))

        writeobj(os.path.join(outpath, "layer_{}.obj".format(layer)), layer_v, layer_f)
        print(len(nodes))
        prev = len(nodes)  # Update prev inside the loop
        layer += 1


def return_child(node, nodes):
    # Function to return child nodes of a BVHNode
    new_nodes = []
    for i in nodes:
        child1, child2 = node[i].child1, node[i].child2
        if child1 != -1:
            new_nodes.append(child1)
        if child2 != -1:
            new_nodes.append(child2)
        if child1 == -1 and child2 == -1:
            new_nodes.append(i)
    return new_nodes


def main():
    print("1. loading mesh... ")
    vertices, triangles = loadobj(argvs[1])
    face_normals, vertices_normals = compute_normal(vertices, triangles)
    print("done")

    print("2. building BVH... ")
    vertices_bvh = np.copy(vertices[triangles.ravel()])
    vertices_bvh_normals = np.copy(vertices_normals[triangles.ravel()])
    root = buildBVH(vertices_bvh.reshape((-1, 3, 3)), vertices_bvh_normals.reshape((-1, 3, 3)),
                    np.arange(triangles.shape[0]))
    print("done")

    print("3. Serializing bvh nodes... ")
    bvhnodelist, bvhtrilist, bvhnormallist, bvhtriindlist = BVHserializer(root)
    np.savez(os.path.join(outpath, f"{os.path.basename(meshpath)}_precomp.npz"),
             vertices=vertices,
             triangles=triangles,
             vertices_bvh=vertices_bvh,
             bvhnodelist=bvhnodelist,
             bvhtrilist=bvhtrilist,
             bvhtriindlist=bvhtriindlist,
             bvhnormallist=bvhnormallist)
    print("done")

    save_each_layer(bvhnodelist)


if __name__ == '__main__':
    argvs = sys.argv
    meshpath, outpath = argvs[1], argvs[2]
    main()