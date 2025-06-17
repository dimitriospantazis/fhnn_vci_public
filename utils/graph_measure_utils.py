import numpy as np
import math
import torch
from manifolds import Lorentz

manifold = Lorentz(k=1.0)
R = 2.0
T= 1.0
HYPERBOLIC_DIMENSION = 3
def compute_hyperbolic_degree_centrality(embeddings):
    # Compute a distance matrix of all distances between embeddings
    dist_matrix = torch.zeros((len(embeddings), len(embeddings)))
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            print(i, j)
            dist_matrix[i, j] = manifold.dist(torch.Tensor(embeddings[i]), torch.Tensor(embeddings[j]))
            dist_matrix[j, i] = dist_matrix[i, j]
    def fermi_dirac_hyp_dc_variant(dist_matrix):
        # probs = 1. / (torch.exp(1 / 2 * ((dist - self.r) / self.t).clamp_max(50.)) + 1.0)
        probs = 1. / (torch.exp(1 / 2 * ((dist_matrix - R) / T)) + 1.0) # NOTE: might need to clamp
        return probs
    fd_prob_matrix = fermi_dirac_hyp_dc_variant(dist_matrix)
    # fd_prob_matrix = fd_prob_matrix
    # Zero out diagonal entries in fd_prob_matrix, we do not want count the probability of a node being connected to itself
    for i in range(len(fd_prob_matrix)):
        fd_prob_matrix[i, i] = 0
    # np.fill_diagonal(fd_prob_matrix, 0)
    # Sum up all probabilities for each node
    node_probs = torch.sum(fd_prob_matrix, dim=1)
    return node_probs


def hyperBC(adjMatrix, coordinatesMatrix):
    # coordinatesMatrix should be the embeddings
    # Pre-Processing of Adjacency Matrix
    nodesNumber = adjMatrix.shape[1]
    pSize = np.max(np.sum(adjMatrix, axis=1))
    pSize = int(pSize)
    nodesNumber = int(nodesNumber)
    testMatrix = np.zeros((int(nodesNumber), int(pSize)), dtype=int)
    indexMatrix = np.zeros(nodesNumber, dtype=int)

    for i in range(nodesNumber):
        tempArray = np.where(adjMatrix[i, :] != 0)[0]
        for j in range(len(tempArray)):
            testMatrix[i, j] = tempArray[j]
        indexMatrix[i] = len(tempArray)

    # RBC initialization
    nodesNumber = coordinatesMatrix.shape[0]
    HBC = np.zeros(nodesNumber)

    for destination in range(5):
        indexP = np.zeros(nodesNumber, dtype=int)
        sigma = np.zeros(nodesNumber)
        P = np.zeros((int(nodesNumber), int(pSize)), dtype=int)  # Initialize P matrix
        sigma[destination] = 1
        distances = np.zeros(nodesNumber)

        # STAGE 1 - TOPOLOGICAL SORT
        dst = coordinatesMatrix[destination, :]
        ysum = 1  # Sxi^2
        
        for j in range(1, HYPERBOLIC_DIMENSION):
            ysum = ysum + dst[j] ** 2

        for vertex in range(nodesNumber):
            xsum = 1  # Syi^2
            xysum = 0

            for j in range(1, HYPERBOLIC_DIMENSION):
                xsum = xsum + coordinatesMatrix[vertex, j] ** 2
                xysum = xysum + coordinatesMatrix[vertex, j] * dst[j]

            t = math.acosh(ysum * xsum - xysum)
            dist = t
            distances[vertex] = dist

        DAG = np.argsort(distances)[::-1]

        # PART 2
        for i in range(nodesNumber - 1, -1, -1):
            v = DAG[i]
            for j in range(indexMatrix[v]):
                w = testMatrix[v, j]
                if distances[w] > distances[v] + 0.3:
                    sigma[w] = sigma[w] + sigma[v]
                    indexP[w] = indexP[w] + 1
                    P[w, indexP[w]] = v

        # PART 3
        delta = np.zeros(nodesNumber)
        for node in range(nodesNumber):
            w = DAG[node]
            if sigma[w] > 0:
                for j in range(indexP[w]):
                    v = P[w, j]
                    delta[v] = delta[v] + (sigma[v] / sigma[w]) * (1 + delta[w])

            if w != destination:
                HBC[w] = HBC[w] + delta[w]

    HBC = HBC / nodesNumber

    return HBC


def htlc(adjMatrix, coordinatesMatrix):
    # Pre-Processing of Adjacency Matrix
    nodesNumber = adjMatrix.shape[1]
    pSize = np.max(np.sum(adjMatrix, axis=1))
    pSize = int(pSize)
    nodesNumber = int(nodesNumber)
    testMatrix = np.zeros((nodesNumber, pSize), dtype=int)
    indexMatrix = np.zeros(nodesNumber, dtype=int)

    for i in range(nodesNumber):
        tempArray = np.where(adjMatrix[i, :] != 0)[0]
        for j in range(len(tempArray)):
            testMatrix[i, j] = tempArray[j]
        indexMatrix[i] = len(tempArray)

    # RBC initialization
    dimensions = coordinatesMatrix.shape[1]
    successorsVi = np.zeros(pSize, dtype=int)
    nodesNumber = coordinatesMatrix.shape[0]
    RBC = np.zeros(nodesNumber)

    for destination in range(nodesNumber):
        distances = np.zeros(nodesNumber)

        # STAGE 1 - TOPOLOGICAL SORT
        dst = coordinatesMatrix[destination, :]
        ysum = 1  # Sxi^2
        for j in range(1, dimensions):
            ysum = ysum + dst[j] ** 2

        for vertex in range(nodesNumber):
            xsum = 1  # Syi^2
            xysum = 0

            for j in range(1, dimensions):
                xsum = xsum + coordinatesMatrix[vertex, j] ** 2
                xysum = xysum + coordinatesMatrix[vertex, j] * dst[j]

            t = math.sqrt(ysum * xsum) - xysum
            if abs(t - 1.0) < 0.001: t = 1.0 # NOTE: acosh() distance might have to be set to 0 for values less than 1
            dist = math.acosh(t)
            distances[vertex] = dist

        DAG = np.argsort(distances)[::-1]

        # STAGE 2 - INIT DELTA
        delta = np.ones(nodesNumber)
        
        # STAGE 3 - ACCUMULATE d.,.(v)
        for i in range(nodesNumber):
            vi = DAG[i]
            distanceVi = distances[vi]
            
            sizeofSucVi = 0
            for j in range(indexMatrix[vi]):
                vj = testMatrix[vi, j]
                if distances[vj] < distanceVi:
                    sizeofSucVi = sizeofSucVi + 1
                    successorsVi[sizeofSucVi] = vj
            
            # Set the value of R(vi,vj)
            if sizeofSucVi != 0:
                R = 1 / sizeofSucVi
            else:
                R = 0
            
            for j in range(1, sizeofSucVi + 1):
                vj = successorsVi[j]
                delta[vj] = delta[vj] + delta[vi] * R
        
        RBC = RBC + delta

    RBC = RBC / nodesNumber

    return RBC
