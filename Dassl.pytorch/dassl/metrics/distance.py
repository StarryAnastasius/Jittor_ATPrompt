# """
# Source: https://github.com/KaiyangZhou/deep-person-reid
# """
# import torch
# from torch.nn import functional as F
#
#
# def compute_distance_matrix(input1, input2, metric="euclidean"):
#     """A wrapper function for computing distance matrix.
#
#     Each input matrix has the shape (n_data, feature_dim).
#
#     Args:
#         input1 (torch.Tensor): 2-D feature matrix.
#         input2 (torch.Tensor): 2-D feature matrix.
#         metric (str, optional): "euclidean" or "cosine".
#             Default is "euclidean".
#
#     Returns:
#         torch.Tensor: distance matrix.
#     """
#     # check input
#     assert isinstance(input1, torch.Tensor)
#     assert isinstance(input2, torch.Tensor)
#     assert input1.dim() == 2, "Expected 2-D tensor, but got {}-D".format(
#         input1.dim()
#     )
#     assert input2.dim() == 2, "Expected 2-D tensor, but got {}-D".format(
#         input2.dim()
#     )
#     assert input1.size(1) == input2.size(1)
#
#     if metric == "euclidean":
#         distmat = euclidean_squared_distance(input1, input2)
#     elif metric == "cosine":
#         distmat = cosine_distance(input1, input2)
#     else:
#         raise ValueError(
#             "Unknown distance metric: {}. "
#             'Please choose either "euclidean" or "cosine"'.format(metric)
#         )
#
#     return distmat
#
#
# def euclidean_squared_distance(input1, input2):
#     """Computes euclidean squared distance.
#
#     Args:
#         input1 (torch.Tensor): 2-D feature matrix.
#         input2 (torch.Tensor): 2-D feature matrix.
#
#     Returns:
#         torch.Tensor: distance matrix.
#     """
#     m, n = input1.size(0), input2.size(0)
#     mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
#     mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#     distmat = mat1 + mat2
#     distmat.addmm_(1, -2, input1, input2.t())
#     return distmat
#
#
# def cosine_distance(input1, input2):
#     """Computes cosine distance.
#
#     Args:
#         input1 (torch.Tensor): 2-D feature matrix.
#         input2 (torch.Tensor): 2-D feature matrix.
#
#     Returns:
#         torch.Tensor: distance matrix.
#     """
#     input1_normed = F.normalize(input1, p=2, dim=1)
#     input2_normed = F.normalize(input2, p=2, dim=1)
#     distmat = 1 - torch.mm(input1_normed, input2_normed.t())
#     return distmat


# """
# Source: https://github.com/KaiyangZhou/deep-person-reid (adapted for Jittor)
# """
import jittor as jt
import jittor.nn as nn


def compute_distance_matrix(input1, input2, metric="euclidean"):
    """A wrapper function for computing distance matrix.

    Each input matrix has the shape (n_data, feature_dim).

    Args:
        input1 (jt.Var): 2-D feature matrix.
        input2 (jt.Var): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".

    Returns:
        jt.Var: distance matrix.
    """
    # Check input
    assert isinstance(input1, jt.Var)
    assert isinstance(input2, jt.Var)
    assert input1.dim() == 2, f"Expected 2-D tensor, but got {input1.dim()}-D"
    assert input2.dim() == 2, f"Expected 2-D tensor, but got {input2.dim()}-D"
    assert input1.size(1) == input2.size(1), "Feature dimensions must match"

    if metric == "euclidean":
        distmat = euclidean_squared_distance(input1, input2)
    elif metric == "cosine":
        distmat = cosine_distance(input1, input2)
    else:
        raise ValueError(
            f"Unknown distance metric: {metric}. "
            'Please choose either "euclidean" or "cosine"'
        )

    return distmat


def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.

    Args:
        input1 (jt.Var): 2-D feature matrix (shape: [m, d]).
        input2 (jt.Var): 2-D feature matrix (shape: [n, d]).

    Returns:
        jt.Var: distance matrix (shape: [m, n]).
    """
    m, n = input1.size(0), input2.size(0)
    # Compute ||input1||^2 (shape: [m, 1] -> expand to [m, n])
    mat1 = jt.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
    # Compute ||input2||^2 (shape: [n, 1] -> transpose to [1, n] -> expand to [m, n])
    mat2 = jt.pow(input2, 2).sum(dim=1, keepdim=True).transpose(0, 1).expand(m, n)
    # Euclidean squared distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2ab
    distmat = mat1 + mat2 - 2 * jt.matmul(input1, input2.transpose(0, 1))
    return distmat


def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (jt.Var): 2-D feature matrix (shape: [m, d]).
        input2 (jt.Var): 2-D feature matrix (shape: [n, d]).

    Returns:
        jt.Var: distance matrix (shape: [m, n]), where distance = 1 - cosine_similarity.
    """
    # Normalize features to unit L2 norm
    input1_normed = nn.normalize(input1, p=2, dim=1)
    input2_normed = nn.normalize(input2, p=2, dim=1)
    # Cosine similarity = input1_normed * input2_normed^T
    # Cosine distance = 1 - cosine_similarity
    distmat = 1 - jt.matmul(input1_normed, input2_normed.transpose(0, 1))
    return distmat


# # 确保这些函数能被外部导入
# __all__ = [
#     "compute_distance_matrix",
#     "euclidean_squared_distance",
#     "cosine_distance"
# ]