"""Keep rehearsal gradients below task gradients in every trainable scope."""
import math

import torch


def preservation_scale(task, preservation, groups, *, maximum=.1, ratio=.5):
    if (not math.isfinite(maximum) or not math.isfinite(ratio)
            or maximum<=0 or not 0<ratio<=1 or not groups):
        raise ValueError('Nonempty parameter groups and finite positive limits required')
    parameters = [p for group in groups for p in group['params']]
    if (not parameters or len({id(p) for p in parameters})!=len(parameters)
            or any(not group['params'] for group in groups)):
        raise ValueError('Disjoint nonempty trainable scopes required')
    derivatives = []
    for loss in [task, preservation]:
        if loss.ndim or not torch.isfinite(loss):
            raise ValueError('Finite scalar objectives required')
        derivatives.append(torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True))
    records, start, weight = [], 0, maximum
    for group in groups:
        stop = start + len(group['params'])
        task_g, replay_g = [row[start:stop] for row in derivatives]
        nt, nr = [math.sqrt(sum(float(g.detach().square().sum()) for g in row if g is not None))
                  for row in [task_g, replay_g]]
        dot = sum(float((a.detach()*b.detach()).sum()) for a,b in zip(task_g,replay_g)
                  if a is not None and b is not None)
        if not all(math.isfinite(v) for v in [nt,nr,dot]):
            raise ValueError('Nonfinite group gradient')
        if nr>0:
            weight = min(weight, ratio*nt/nr)
        records.append(dict(group=group['name'], task_gradient_norm=nt,
            preservation_gradient_norm=nr, gradient_cosine=dot/max(nt*nr,1e-30)))
        start = stop
    for row in records:
        row['weighted_fraction'] = weight*row['preservation_gradient_norm']/max(row['task_gradient_norm'],1e-30)
        if row['weighted_fraction']>ratio+1e-10:
            raise ValueError('A trainable scope escaped the declared gradient cap')
    return weight, dict(per_group=records,
        maximum_weighted_gradient_fraction=max(r['weighted_fraction'] for r in records))
