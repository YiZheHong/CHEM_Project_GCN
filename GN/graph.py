import math
import numpy as np

from .node import Node
class Graph:
    def __init__(self, list,max_node):
        self.max = max_node
        self.num_of_chem = int(list[0][2])
        self.num_of_pseudo = max_node - self.num_of_chem
        self.charge = int(list[1])
        self.y_value = float(list[2])
        self.M = int(list[3])
        self.Vib = float(list[4])
        self.Dis = float(list[5])
        self.chems = []
        self.edges = []
        self.sum_dist = 0
        self.A_sc = np.zeros((max_node,max_node),dtype=np.float32)
        self.A = np.zeros((max_node,max_node),dtype=np.float32)
        for i in range(int(self.num_of_chem)):
            chem = list[6+i][1:len(list[6+i])-1].replace("'",'').replace(" ",'').split(',')
            self.chems.append(self.make_Chems(chem,i))
        for i in range(0,self.num_of_pseudo):
            chem = ['Pse','0','0','0']
            self.chems.append(self.make_Chems(chem,self.num_of_chem+i))
        for a in self.chems:
            if a.element != "Pse":
                for b in self.chems:
                    if b.element != "Pse":
                        if id(a) != id(b):
                            edge = self.make_edges(a,b)
                            if edge!= None and self.containEdge(edge)==False:
                                self.edges.append(edge)
                                self.sum_dist += edge[2]
        self.make_A_matrix()
        self.make_A_sc_matrix()
        self.num_edges = len(self.edges)

    def containEdge(self,e1):
        for e2 in self.edges:
            if e1[0] == e2[1] and e1[1]==e2[0]:
                return True
        return False
    def make_Chems(self,list,num):
        chem = Node(list,num)
        return chem
    def distance(self,a,b):
        p1 = np.array([a.x, a.y, a.z])
        p2 = np.array([b.x, b.y, b.z])
        squared_dist = np.sum((p1 - p2) ** 2, axis=0)
        dist = np.sqrt(squared_dist)
        return dist
    
    def make_edges(self,a,b):
        dist = self.distance(a,b)
        if dist<=3:
            edge = [a.num, b.num,dist]
            return edge
        return None
    
    def make_A_sc_matrix(self):
        for edge in self.edges:
            if edge[0] != edge[1]:
                self.A_sc[edge[0], edge[1]] = self.similarity(edge)
                self.A_sc[edge[1], edge[0]] = self.similarity(edge)
        for i in range(self.num_of_chem):
            self.A_sc[i, i] = 1


    def make_A_matrix(self):
        for edge in self.edges:
            if edge[0] != edge[1]:
                self.A[edge[0],edge[1]]=self.similarity(edge)
                self.A[edge[1], edge[0]] = self.similarity(edge)



    def similarity(self,edge):
        dis = float(edge[2])
        if dis == 0:
            return 1
        return round(1/(dis**2),2)

    def __str__(self):
        return str(self.y_value)+', '+str(self.charge)+', '+str(self.num_of_chem)
    def chems_toString(self):
       s = []
       for i in self.chems:
           s.append(i.element)

       return str(s)

