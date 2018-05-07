# slice and chop spn
import numpy as np
import math
from ProductNode import ProductNode
from Node import LeafNode
from SumNode import SumNode
import copy
import collections

import random

class SPN():

    LOG_ZERO = float('-inf')

    def __init__(self):
        self._curr_node_id = -1
        self.nodes = []


    ###############################################################
    #               Generative gradient descent                   #
    ###############################################################    

    def evaluate(self,evidence):

        self.set_leaves(evidence)
        root = self.get_root()
        return root.evaluate(self)


    def update_weights(self, N, learning_rate):
        for node in self.get_sum_nodes():
            mean = 0
            if len(node.children) > 0:
                for child_id in node.children.keys():
                    mean += node.children[child_id][1]/N
                mean = mean / (len(node.children))

            for child_id in node.children.keys():
                
                avg_gradient =  node.children[child_id][1] / N
                print("update for", node.id, "to", child_id, "is", avg_gradient)
                update_value = avg_gradient - mean

                # Projected gradient update
                #node.children[child_id][0] += learning_rate * update_value
                # Exponential gradient update
                node.children[child_id][0] *= np.exp(learning_rate * update_value)
                # reset accumulationg derivatives
                node.children[child_id][1] = 0

                # dont let weights go negative
                if node.children[child_id][0] < 0:
                    node.children[child_id][0] = 0

        self.normalise_weights()

    def generative_soft_gd(self, data, batch=True, learning_rate=0.1):
        
        for instance in data:  
            output = self.evaluate(instance)

            self.clear_derivatives()
            root = self.get_root()
            
            # dll_droot = 1/output, maximise log likelihood: -log(output)
            if output == SPN.LOG_ZERO:
                continue
            root.logDerivative = -output

            root.backprop(self)
            
            if batch == False:
                self.update_weights(1, learning_rate)

        if batch == True:
            self.update_weights(len(data), learning_rate)

    ###############################################################
    #             Generative hard gradient descent                #
    ###############################################################    

    def evaluate_mpn(self,evidence):

        self.set_leaves(evidence)
        root = self.get_root()
        return root.evaluate_mpn(self)


    def generative_hard_gd(self, data, batch=True, learning_rate=0.1):

        for instance in data:  

            self.clear_derivatives()

            output = self.evaluate_mpn(instance)
            root = self.get_root()
            if output == SPN.LOG_ZERO:
                # this only causes problems with XOR example
                continue
            else:
                root.logDerivative = -output

            root.hard_backprop(self)

            # update weights online
            if batch == False:
                self.update_weights_gen_hard(1, learning_rate=0.1)

        # upadte weight batch
        if batch == True:
            self.update_weights_gen_hard(len(data), learning_rate=0.05)
        self.normalise_weights()


    def update_weights_gen_hard(self, N, learning_rate):

        for node in self.get_sum_nodes():
            for child_id in node.children.keys():
                # update the weight
                node.children[child_id][0] += (node.children[child_id][1]/N)*learning_rate
                # clear the counts holder
                node.children[child_id][1] = 0

    ###############################################################
    #          Discriminitive soft gradient descent               #
    ###############################################################  

    def discriminitive_soft_gd(self, data, labels, batch=True, learning_rate=0.1):

        copy_spn = copy.deepcopy(self)

        for instance, y in zip(data, labels): 
            # reset derivatives 
            self.clear_derivatives()
            copy_spn.clear_derivatives()

            # correct label
            inst = np.append(instance,y)
            output = self.evaluate(inst)
            root = self.get_root()

            if output == SPN.LOG_ZERO:
                continue
            else:
                root.logDerivative = -output
            root.backprop(self)


            # best guess
            output = copy_spn.best_guess(instance) 
            root2 = copy_spn.get_root()
            if output == SPN.LOG_ZERO:
                continue
            else:
                root2.logDerivative = -output
            root2.backprop(self)

        if batch == True:
            # update weights
            self.disc_update_weights_soft(copy_spn, learning_rate, N=len(data))
            self.normalise_weights()


    def disc_update_weights_soft(self, best_guess, learning_rate, N=1):
        for node, guess in zip(self.get_sum_nodes(), best_guess.get_sum_nodes()):
            # get means
            mean = 0 
            for child_id, guess_child_id in zip(node.children.keys(), guess.children.keys()):
                # mean of differences
                tmp = node.children[child_id][1] - guess.children[guess_child_id][1]
                mean += tmp
            mean = mean / N

            # update weights
            for child_id, guess_child_id in zip(node.children.keys(), guess.children.keys()):
                # difference
                tmp = node.children[child_id][1] - guess.children[guess_child_id][1]
                print("id", child_id, "difference", tmp/N)
                print("part", node.children[child_id][1]/N)
                
                # take mean out
                update_value = tmp/N - mean
                print("update from",node.id,"to", child_id, "without mean", tmp/N)
                #node.children[child_id][0] += tmp*learning_rate/N
                # Exponential gradient update
                node.children[child_id][0] *= np.exp(learning_rate * update_value)

            # clear counts
            for child_id in node.children.keys():
                node.children[child_id][1] = 0


    def best_guess(self, evidence):
        for leaf in self.get_leaves():
            var_value = 0

            # evidence fill leaves normally
            if leaf.order < len(evidence):
                var_value = evidence[leaf.order]            
                if leaf.inverse:
                    var_value = 1 - var_value
            # marginalise label
            else:
                var_value = 1
            leaf.setValue(var_value)

        root = self.get_root()
        value = root.evaluate(self)

        return value


    ###############################################################
    #          Discriminitive hard gradient descent               #
    ###############################################################    

    def best_guess_hard(self, evidence):
        for leaf in self.get_leaves():
            var_value = 0

            # evidence fill leaves normally
            if leaf.order < len(evidence):
                var_value = evidence[leaf.order]            
                if leaf.inverse:
                    var_value = 1 - var_value
            # marginalise label
            else:
                var_value = 1
            leaf.setValue(var_value)

        root = self.get_root()
        value = root.evaluate_mpn(self)

        return value


    def discriminitive_hard_gd(self, data, labels, batch=True, learning_rate=0.1):
        
        copy_spn = copy.deepcopy(self)

        for instance, y in zip(data, labels):  
            self.clear_derivatives()
            copy_spn.clear_derivatives()

            # correct label
            inst = np.append(instance,y)
            output = self.evaluate_mpn(inst)
            root = self.get_root()
            root.logDerivative = -output 
            root.hard_backprop(self)

            # best guess
            copy_spn.best_guess_hard(instance) 
            root = copy_spn.get_root()
            root.hard_backprop(self)


        if batch == True:
            # update weights
            self.disc_update_weights_hard(copy_spn, learning_rate)
            self.normalise_weights()


    def disc_update_weights_hard(self, best_guess, learning_rate):
        for node, guess in zip(self.get_sum_nodes(), best_guess.get_sum_nodes()):

            for child_id, guess_child_id in zip(node.children.keys(), guess.children.keys()):
                # counts difference
                tmp = node.children[child_id][1] - guess.children[guess_child_id][1]
                #print("id", child_id, "difference", tmp)
                if tmp > 0:
                    node.children[child_id][0] += tmp*learning_rate

            # clear counts
            for child_id in node.children.keys():
                node.children[child_id][1] = 0


    ###############################################################
    #                              Utils                          #
    ###############################################################  

    def get_next_node_id(self):
        self._curr_node_id += 1
        return self._curr_node_id


    def get_node_by_id(self, node_id):
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise ValueError("Node id '{}' not found in SPN".format(node_id))


    def get_root(self):
        return self.get_node_by_id(0)


    def add_node(self,node, weight=1):
        self.nodes.append(node)
        # not for the root node id 0
        if node.id != 0:

            # check if the the noe has many parents and iterate through them
            if isinstance(node.parent_id, collections.Iterable):
                j = 0
                for i in node.parent_id:
                    parent = self.get_node_by_id(i)
                    if isinstance(weight, collections.Iterable):
                        parent.add_child(node.id, weight[j])
                    else:
                        parent.add_child(node.id, weight)
                    j += 1
            else:
                # or just add link to one parent
                parent = self.get_node_by_id(node.parent_id)
                parent.add_child(node.id, weight)


    def get_leaves(self):
        return [n for n in self.nodes if isinstance(n,LeafNode)]


    def get_sum_nodes(self):
        return [n for n in self.nodes if isinstance(n,SumNode)]


    def init_children(self):
        for node in self.nodes:
            x = len(node.children)
            while x > 0:
                for i in node.children: 
                    node.children

    def set_leaves(self, evidence):
        for leaf in self.get_leaves():
            var_value = evidence[leaf.order]
            
            # for not x1 situations
            if leaf.inverse:
                var_value = 1 - var_value
            leaf.setValue(var_value)

    def clear_derivatives(self):
        for node in self.nodes:
            node.logDerivative = SPN.LOG_ZERO


    def compute_marginal(self):
        for leaf in self.get_leaves():
            leaf.setValue(1)

        root = self.get_root()
        value = root.evaluate(self)
        return value


    def normalise_weights(self):
        for node in self.get_sum_nodes():
            # sum of that sum nodes' weights
            z = 0
            for child_id in node.children.keys():
                z += node.children[child_id][0]
            if z == 0:
                return

            for child_id in node.children.keys():
                node.children[child_id][0] = node.children[child_id][0] / z


    def map_inference(self, evidence):
        # what is the probability y = 1
        # numerator joint p(x1,x2,y)
        denom_value = self.evaluate(np.append(evidence, 1))

        # denominator p(x1,x2)
        num_value = self.best_guess(evidence)

        if num_value == SPN.LOG_ZERO:
            return SPN.LOG_ZERO
        conditional = num_value - denom_value
        return conditional


    def query(self, evidence):
        # compute marginals
        x = []
        value = self.best_guess(evidence)
        x.append(value)


        # numerators
        x1 = self.evaluate(np.append(evidence, 1))
        x2 = self.evaluate(np.append(evidence, 0))

        x.append(x1-x[0])
        x.append(x2-x[0])
        return np.exp(x)


    def print_weights(self):
        for node in self.get_sum_nodes():
            print("\nNode ID", node.id)
            for child_id in node.children.keys():
                print("To node", child_id, ":    %.4f" % node.children[child_id][0])
