"""Training-only conditional position supervision, with no runtime controller."""
import torch


def paired_response_loss(prediction,target):
    if (prediction.ndim!=4 or prediction.shape[1:]!=(2,2,3) or target.shape!=prediction.shape
            or len(prediction)==0 or not torch.isfinite(prediction).all() or not torch.isfinite(target).all()):
        raise ValueError('Finite paired same-pose first2 XYZ positions required')
    expected=target[:,1]-target[:,0]
    actual=prediction[:,1]-prediction[:,0]
    scale=torch.linalg.vector_norm(expected,dim=-1,keepdim=True).clamp_min(.00005)
    return ((actual-expected)/scale).square().mean()


def worst_prefix_position_loss(prediction,target):
    if (prediction.ndim!=3 or prediction.shape[1:]!=(2,3) or target.shape!=prediction.shape
            or len(prediction)==0 or not torch.isfinite(prediction).all() or not torch.isfinite(target).all()):
        raise ValueError('Finite nonempty executable first2 XYZ positions required')
    return torch.linalg.vector_norm(prediction-target,dim=-1).max()/.0001
