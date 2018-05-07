import numpy as np
import torch
from torch.autograd import Variable


class Node():

    LOG_ZERO = Variable(torch.tensor(float('-inf')), requires_grad=True)
    if torch.cuda.is_available():
        LOG_ZERO = Variable(torch.tensor(float('-inf')), requires_grad=True).cuda()

    def __init__(self, id, parent_id, children=[]):
        self.id = id
        self.children = children
        self.parent_id = parent_id
        self.logValue = Node.LOG_ZERO
        self.logDerivative = Node.LOG_ZERO

    def setLogValue(self, logValue):
        self.value = torch.tensor(logValue)

    def log_sum_exp(a, b):
        expsum = torch.exp(a) + torch.exp(b)
        return torch.log(expsum)


class LeafNode(Node):

    def __init__(self, id, parent_id, order, inverse, value=1):
        self.id = id
        self.parent_id = parent_id
        self.value = torch.tensor(1.0)
        self.logValue = torch.tensor(0)
        self.logDerivative = Node.LOG_ZERO
        self.order = order  # for feeding in evidence
        self.inverse = inverse

    def setValue(self, value):
        if torch.cuda.is_available():
            if value == torch.tensor(0.0).cuda():
                self.logValue = Node.LOG_ZERO
            else:
                self.logValue = torch.tensor(np.log(value)).cuda()
                #self.logValue = value
        else:
            if value == torch.tensor(0.0):
                self.logValue = Node.LOG_ZERO
            else:
                self.logValue = torch.tensor(np.log(value))
                #self.logValue = value

    def evaluate(self, spn):
        return self.logValue

    def evaluate_mpn(self, spn):
        return self.logValue

    def hard_backprop(self, spn):
        # do nothing for leaf node
        pass

    def backprop(self, spn):
        # do nothing for leaf node
        pass
