import numpy as np
from Node import Node


class SumNode(Node):
    """
        Sum node of SPN
    """

    def __init__(self, id, parent_id):
        super().__init__(id, parent_id)
        self.children = dict()

    def add_child(self, child_id, weight):
        self.children.update({child_id: [weight, 0]})

    ###############################################################
    #                     gradient descent                        #
    ###############################################################

    def evaluate(self, spn):

        weighted_children = Node.LOG_ZERO
        for child_id in self.children.keys():

            if self.children[child_id][0] == 0:
                continue

            # taking the log of the weight + log val
            weighted_child = spn.get_node_by_id(child_id).evaluate(spn) + np.log(self.children[child_id][0])
            if weighted_child == Node.LOG_ZERO:
                continue

            weighted_children = np.logaddexp(weighted_children, weighted_child)
        self.logValue = weighted_children
        return weighted_children

    def backprop(self, spn):
        # unused parent, dont bother passing derivative
        if self.logDerivative == Node.LOG_ZERO:
            #print("unused parent derivative sum", self.id)
            return
        if self.logValue == Node.LOG_ZERO:
            #print("unused parent value sum", self.id)
            return

        for child_id in self.children.keys():

            child = spn.get_node_by_id(child_id)

            # backprop from parent
            if self.children[child_id][0] == 0:
                temp = Node.LOG_ZERO
            else:
                # parent derivative times weight
                temp = self.logDerivative + np.log(self.children[child_id][0])

            # pass derivative
            if child.logDerivative == Node.LOG_ZERO:
                child.logDerivative = temp
            else:
                # accumulate
                child.logDerivative = np.logaddexp(temp, child.logDerivative)

            # update the values
            if child.logValue == Node.LOG_ZERO:
                update = Node.LOG_ZERO
            else:
                update = child.logValue + self.logDerivative
                #print("ds_dw child val", child_id, np.exp(child.logValue), "parent",self.id," derivative", np.exp(self.logDerivative))
                #print("Update val",  np.exp(update))

            # ds_dw
            self.children[child_id][1] += np.exp(update)

            #print("passing derivative", np.exp(child.logDerivative), "from", self.id, "to", child.id)
            child.backprop(spn)

    ###############################################################
    #                    hard gradient descent                    #
    ###############################################################

    def hard_backprop(self, spn):
        # unused parent, goes here when upward pass did not pass through here
        if self.logValue == Node.LOG_ZERO:
            return

        for child_id in self.children.keys():
            child = spn.get_node_by_id(child_id)

            # if one of the children are zero the product is zero
            if child.logValue == Node.LOG_ZERO:
                continue
            else:
                # increment counts
                self.children[child_id][1] += 1
            child.hard_backprop(spn)

    def evaluate_mpn(self, spn):

        max_child = Node.LOG_ZERO
        for child_id in self.children.keys():
            if self.children[child_id][0] == 0:
                continue
            else:
                child = spn.get_node_by_id(child_id).evaluate(spn) + np.log(self.children[child_id][0])
            if child > max_child:
                max_child = child
        self.logValue = max_child
        return max_child
