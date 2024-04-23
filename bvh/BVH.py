import numpy as np
from collections import deque

class AABB:
    def __init__(self, botleft, topright):
        self.botleft = botleft
        self.topright = topright

class BVHNode:
    def __init__(self, aabb, triangles, vernormals, triindices, child1=None, child2=None):
        self.aabb = aabb
        self.triangles = triangles
        self.vernormals = vernormals
        self.triindices = triindices
        self.child1 = child1
        self.child2 = child2
        
def buildBVH(triangles, vernormals, triindices):
    if triangles.shape[0] == 0:  # Check if triangles array is empty
        # If triangles array is empty, return a BVHNode with empty arrays
        return BVHNode(AABB(np.zeros(3), np.zeros(3)), np.zeros((0, 3, 3), dtype=np.float32),
                       np.zeros((0, 3, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32), None, None)

    # Compute AABB
    aabb = AABB(np.min(triangles, axis=0), np.max(triangles, axis=0))

    if triangles.shape[0] <= 64:
        # If number of triangles is less than or equal to 64, return a BVHNode with current triangles
        return BVHNode(aabb, triangles, vernormals, triindices, None, None)
    else:
        # Set variables
        mincost = np.inf
        minaxis = -1
        minsplit = -1
        centroids = np.mean(triangles, axis=1)

        for axis in range(3):
            # Sort triangles with respect to centroids
            sorted_tris = triangles[np.argsort(centroids[:, axis])]

            # Calculate left AABB
            left_aabb = AABB(np.inf, -np.inf)
            left_cost = np.zeros((len(sorted_tris),), dtype=np.float32)
            for i, tri in enumerate(sorted_tris):
                left_aabb.botleft = np.minimum(left_aabb.botleft, tri.min(axis=0))
                left_aabb.topright = np.maximum(left_aabb.topright, tri.max(axis=0))
                diag = np.abs(left_aabb.topright - left_aabb.botleft)
                left_index = i
                left_cost[left_index] = (diag[0] * diag[1] + diag[1] * diag[2] + diag[2] * diag[0]) * i

            # Calculate right AABB
            right_aabb = AABB(np.inf, -np.inf)
            right_cost = np.zeros((len(sorted_tris),), dtype=np.float32)
            for i, tri in enumerate(sorted_tris[::-1]):
                right_aabb.botleft = np.minimum(right_aabb.botleft, tri.min(axis=0))
                right_aabb.topright = np.maximum(right_aabb.topright, tri.max(axis=0))
                diag = np.abs(right_aabb.topright - right_aabb.botleft)
                right_index = len(sorted_tris) - 1 - i
                right_cost[right_index] = (diag[0] * diag[1] + diag[1] * diag[2] + diag[2] * diag[0]) * i

            # Update minimum cost, axis, index
            if (left_cost + right_cost).min() < mincost:
                mincost = (left_cost + right_cost).min()
                minaxis = axis
                minsplit = np.argmin(left_cost + right_cost)

        sorted_indices = np.argsort(centroids[:, minaxis])
        sorted_triindices = triindices[sorted_indices]
        sorted_tris = triangles[sorted_indices]
        sorted_vern = vernormals[sorted_indices]
        node1 = buildBVH(sorted_tris[:minsplit], sorted_vern[:minsplit], sorted_triindices[:minsplit])
        node2 = buildBVH(sorted_tris[minsplit:], sorted_vern[minsplit:], sorted_triindices[minsplit:])

        return BVHNode(aabb,
                       np.zeros((0, 3, 3), dtype=np.float32),
                       np.zeros((0, 3, 3), dtype=np.float32),
                       np.zeros((0,), dtype=np.int32),
                       node1,
                       node2)


class SerialBVHNode:
    def __init__(self, aabb, tristart, ntris):
        self.aabb = aabb
        self.tristart = tristart
        self.ntris = ntris
        self.child1 = -1
        self.child2 = -1

def BVHserializer(root):
    queue = deque()
    nodelist = []  

    queue.append(root)  

    while queue:
        front = queue.popleft()
        nodelist.append(front)    # Add current node to the list

        if front.child1 is not None:
            queue.append(front.child1)  # Add child1 to the queue
        if front.child2 is not None:
            queue.append(front.child2)  # Add child2 to the queue

    for node in nodelist:
        node.parent = None  # Reset parent attribute for each node

    serialnodelist = []
    serialtrilist = np.zeros((0, 3, 3), dtype=np.float32)
    serialnormallist = np.zeros((0, 3, 3), dtype=np.float32)
    serialtriindlist = np.zeros(0, dtype=np.int32)

    for node in nodelist:
        serialnodelist.append(SerialBVHNode(node.aabb, serialtrilist.shape[0], node.triangles.shape[0]))      
        if node.child1 is not None:
            node.child1.parent = serialnodelist[-1]
        if node.child2 is not None:
            node.child2.parent = serialnodelist[-1]      

        serialtrilist = np.vstack((serialtrilist, node.triangles))
        serialnormallist = np.vstack((serialnormallist, node.vernormals))
        serialtriindlist = np.hstack((serialtriindlist, node.triindices))  

    for i, (node, snode) in enumerate(zip(nodelist, serialnodelist)):
        if node.parent is not None:
            if node.parent.child1 == -1:
                node.parent.child1 = i
            elif node.parent.child2 == -1:
                node.parent.child2 = i  

    return serialnodelist, serialtrilist, serialnormallist, serialtriindlist
