# slice and chop spn
import numpy as np
import math
from Node import ProductNode
from Node import LeafNode
from Node import SumNode
import copy
import collections

class SPN():

    LOG_ZERO = float('-inf')

    def __init__(self):
        self._curr_node_id = -1
        self.nodes = []


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


    def evaluate(self,evidence):
        # Feed the leaf node with data point or evidence
        for leaf in self.get_leaves():
            var_value = evidence[leaf.order]
            
            # for not x1 situations
            if leaf.inverse:
                var_value = 1 - var_value
            leaf.setValue(var_value)

        root = self.get_root()
        value = root.evaluate(self)
        return np.exp(value)


    def evaluate_mpn(self,evidence):
        # Feed the leaf node with data point or evidence
        for leaf in self.get_leaves():
            var_value = evidence[leaf.order]
            
            # for not x1 situations
            if leaf.inverse:
                var_value = 1 - var_value
            leaf.setValue(var_value)

        root = self.get_root()
        value = root.evaluate_mpn(self)
        return np.exp(value)


    def update_weights_gen_hard(self):
        for node in self.get_sum_nodes():

            max_id = list(node.children)[0]
            
            for child_id in node.children.keys():
                if node.children[child_id][1] > node.children[max_id][1]:
                    max_id = child_id

            if node.children[max_id][1] != 0:
                node.children[max_id][0] += 1

            # clear aux holder
            for child_id in node.children.keys():
                node.children[child_id][1] = 0


    def update_weights(self, N):
        for node in self.get_sum_nodes():
            lr = 0.1

            for child_id in node.children.keys():
                #print("id", child_id)
                node.children[child_id][0] += lr * (node.children[child_id][1])/N
                node.children[child_id][1] = 0


    def generative_soft_gd(self, data):
        
        for instance in data:  
            output = self.evaluate(instance)


            self.clear_derivatives()
            

            
            if output == 0:
                d_ds = 0
            else:
                d_ds = 1 / output

            #d_ds = 0
            #d_ds = -np.log(output)
            root = self.get_root()
            root.logDerivative = d_ds
            root.backprop(self)

            #self.update_weights(1)
        self.update_weights(len(data))



    def update_weights_hard(self, best_guess):
        for node, guess in zip(self.get_sum_nodes(), best_guess.get_sum_nodes()):

            for child_id, guess_child_id in zip(node.children.keys(), guess.children.keys()):
                tmp = node.children[child_id][1] - guess.children[guess_child_id][1]
                if tmp > 0:
                    node.children[child_id][0] += tmp
            #if node.id == 10:
                #print(node.children[child_id][1])
                #print(guess.children[guess_child_id][1])
            # clear aux holder
            for child_id in node.children.keys():
                node.children[child_id][1] = 0


    def generative_hard_gd(self, data):
        for instance in data:  
            output = self.evaluate(instance)
            root = self.get_root()
            if root.logDerivative == SPN.LOG_ZERO:
                root.logDerivative = output
            else:
                root.logDerivative += output

            root.logDerivative = 0
            root.hard_backprop(self)
            self.update_weights_gen_hard()
            self.clear_derivatives()


    def discriminitive_hard_gd(self, data, labels):
        
        for instance, y in zip(data, labels):  

            copy_spn = copy.deepcopy(self)
            # correct label
            inst = np.append(instance,y)
            self.evaluate_mpn(inst)
            root = self.get_root()
            root.hard_backprop(self)


            # best guess
            copy_spn.best_guess(instance) # not gonna work
            root = copy_spn.get_root()
            root.hard_backprop(self)


        # update weights
            self.update_weights_hard(copy_spn)
        self.clear_derivatives()


            
    def query(self, evidence):
        x = []
        for leaf in self.get_leaves():
            var_value = 0
            if leaf.order < len(evidence):
                var_value = evidence[leaf.order]            
                # for not x1 situations
                if leaf.inverse:
                    var_value = 1 - var_value



            # hacky prob
            else:
                var_value = 1
            leaf.setValue(var_value)

        root = self.get_root()
        value = root.evaluate(self)
        x.append(np.exp(value))


        # numerators
        x1 = self.evaluate(np.append(evidence, 1))
        x2 = self.evaluate(np.append(evidence, 0))

        x.append(x1/x[0])
        x.append(x2/x[0])
        return x


    def best_guess(self, evidence):
        for leaf in self.get_leaves():
            var_value = 0
            if leaf.order < len(evidence):
                var_value = evidence[leaf.order]            
                # for not x1 situations
                if leaf.inverse:
                    var_value = 1 - var_value



            # hacky prob
            else:
                var_value = 1
            leaf.setValue(var_value)

        root = self.get_root()
        value = root.evaluate_mpn(self)

        return value

    def clear_derivatives(self):
        for node in self.nodes:
            node.logDerivative = SPN.LOG_ZERO


    def compute_marginal(self):
        for leaf in self.get_leaves():
            leaf.setValue(1)

        root = self.get_root()
        value = root.evaluate(self)
        return np.exp(value)


    def normalise_weights(self):
        for node in self.get_sum_nodes():
            # sum of that sum nodes' weights
            z = 0
            for child_id in node.children.keys():
                z += node.children[child_id][0]

            for child_id in node.children.keys():
                node.children[child_id][0] = node.children[child_id][0] / z


    def map_inference(self):
        # what is the probability x3 = 0
        # numerator joint p(x1,x2,y)
        evidence = np.array([0.1,0.1,0.0])
        for leaf in self.get_leaves():
            var_value = evidence[leaf.order]
            
            # for not x1 situations
            if leaf.inverse:
                var_value = 1 - var_value
            leaf.setValue(var_value)

        root = self.get_root()
        num_value = root.evaluate(self)


        # denominator p(x1,x2)
        evidence = np.array([0.1,0.1])
        for leaf in self.get_leaves():
            if leaf.order < len(evidence):
                var_value = evidence[leaf.order]
            
                # for not x1 situations
                if leaf.inverse:
                    var_value = 1 - var_value
                leaf.setValue(var_value)
            else:
                leaf.setValue(1.0)

        denom_value = root.evaluate(self)
        print("parts")
        print(num_value)
        print(denom_value)
        print(num_value - denom_value)
        conditional = num_value - denom_value
        return np.exp(conditional)


    def print_weights(self):
        for node in self.get_sum_nodes():
            print("\nNode ID", node.id)
            for child_id in node.children.keys():
                print("To node", child_id, ":    %.4f" % node.children[child_id][0])