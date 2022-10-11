import torch

def dynamic_partition(data, partitions, num_partitions):
    res = []
    for i in range(num_partitions):
      res.append(data[(partitions == i).nonzero().squeeze(1)])
    return res