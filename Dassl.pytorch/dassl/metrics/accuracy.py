import jittor as jt

def compute_accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for
    the specified values of k.
    (Jittor 兼容版本)
    """
    maxk = max(topk)
    batch_size = target.shape[0]

    if isinstance(output, (tuple, list)):
        output = output[0]

    # topk 返回 (values, indices)
    _, pred = jt.topk(output, maxk, dim=1)
    pred = pred.transpose(0, 1)

    target_exp = target.view(1, -1).broadcast(pred.shape)
    correct = jt.equal(pred, target_exp)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float32().sum()
        acc = correct_k * (100.0 / batch_size)
        res.append(acc)
    return res
