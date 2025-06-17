import torch

def get_centroid_regression_distance(node_repr : torch.Tensor, mask : torch.Tensor, centroid_embedding : torch.nn.Embedding, manifold)-> torch.Tensor:
    """
    Compute the centroid regression loss.

    Args:
        node_repr (torch.Tensor): Node representations [node_num, embed_size]
        NUM ROIS x 3 embeddings_list !!
        mask (torch.Tensor): Mask to identify real nodes [node_num, 1]
        centroid_embedding (torch.nn.Embedding): Embedding for centroids
        manifold: Manifold object (hyperbolic or euclidean)

    Returns:
        loss (torch.Tensor): Centroid regression loss
    """
    node_num = node_repr.size(0)

    # Broadcast and reshape node_repr to [node_num * num_centroid, embed_size]
    node_repr = node_repr.unsqueeze(1).expand(
        -1, centroid_embedding.num_embeddings, -1
    ).contiguous().view(-1, centroid_embedding.embedding_dim)

    # Broadcast and reshape centroid embeddings to [node_num * num_centroid, embed_size]
    if manifold == 'hyperbolic':
        centroid_repr = manifold.log_map_zero(centroid_embedding.weight)
    else:
        centroid_repr = centroid_embedding.weight

    centroid_repr = centroid_repr.unsqueeze(0).expand(
        node_num, -1, -1
    ).contiguous().view(-1, centroid_embedding.embedding_dim)

    # Calculate distance
    node_centroid_dist = manifold.dist(node_repr, centroid_repr)
    node_centroid_dist = node_centroid_dist.view(1, node_num, centroid_embedding.num_embeddings) * mask

    # Average pooling over nodes
    graph_centroid_dist = torch.sum(node_centroid_dist, dim=1) / torch.sum(mask)

    return graph_centroid_dist
