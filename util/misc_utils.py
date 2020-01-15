# coding=utf-8
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# def move_to_cuda(sample):
#
#     def _move_to_cuda(maybe_tensor):
#         if torch.is_tensor(maybe_tensor):
#             return maybe_tensor.cuda()
#         elif isinstance(maybe_tensor, dict):
#             return {
#                 key: _move_to_cuda(value)
#                 for key, value in maybe_tensor.items()
#             }
#         elif isinstance(maybe_tensor, list):
#             return [_move_to_cuda(x) for x in maybe_tensor]
#         else:
#             return maybe_tensor
#
#     return _move_to_cuda(sample)

def move_to_cuda(sample):

    for atr in dir(sample):
        value = getattr(sample, atr)
        if torch.is_tensor(value):
            setattr(sample, atr, value.cuda())

    return sample