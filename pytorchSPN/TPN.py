import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch

from Node import Node
from SumNode import SumNode
from ProductNode import ProductNode
from Node import LeafNode
from SPN import SPN
import random

import numpy as np


class TPN(nn.Module):
    def __init__(self):
        super(TPN, self).__init__()
        spn = SPN()

        # root node id = 0
        root_node = SumNode(0, 0)
        spn.add_node(root_node)

        # Product nodes 1 and 2
        prod_node1 = ProductNode(1, 0)
        prod_node2 = ProductNode(2, 0)

        w1 = nn.Parameter(torch.rand(1))
        #w1.register_hook(lambda g: print(g))
        w2 = nn.Parameter(torch.rand(1))
        spn.add_node(prod_node1, w1)
        spn.add_node(prod_node2, w2)

        # Sum nodes 3 and 4
        sum_node3 = SumNode(3, 1)
        sum_node4 = SumNode(4, 2)
        spn.add_node(sum_node3)
        spn.add_node(sum_node4)

        # Product nodes 5-8
        prod_node5 = ProductNode(5, 4)
        prod_node6 = ProductNode(6, 4)
        prod_node7 = ProductNode(7, 3)
        prod_node8 = ProductNode(8, 3)
        w3 = nn.Parameter(torch.rand(1))
        w4 = nn.Parameter(torch.rand(1))
        w5 = nn.Parameter(torch.rand(1))
        w6 = nn.Parameter(torch.rand(1))

        spn.add_node(prod_node5, w3)
        spn.add_node(prod_node6, w4)
        spn.add_node(prod_node7, w5)
        spn.add_node(prod_node8, w6)

        # Add leaf nodes
        X1 = LeafNode(9, np.array([6, 7]), 0, False)
        X_1 = LeafNode(10, np.array([5, 8]), 0, True)
        X2 = LeafNode(11, np.array([5, 7]), 1, False)
        X_2 = LeafNode(12, np.array([6, 8]), 1, True)
        X3 = LeafNode(13, np.array([2]), 2, False)
        X_3 = LeafNode(14, np.array([1]), 2, True)
        spn.add_node(X1)
        spn.add_node(X_1)
        spn.add_node(X2)
        spn.add_node(X_2)
        spn.add_node(X3)
        spn.add_node(X_3)
        spn.normalise_weights()

        self.weights = [w1, w2, w3, w4, w5, w6]
        self.spn = spn

    def forward(self, x):

        self.spn.set_leaves(x)
        root = self.spn.get_root()
        return root.evaluate(self.spn)

        return x

    def print_weights(self):
        # for param in self.weights:
        #    if param.requires_grad:
        #        print(param.name, param.data)
        self.spn.print_weights()
