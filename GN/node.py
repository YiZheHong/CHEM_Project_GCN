class Node:
    def __init__(self, list,num):
        self.num = int(num)
        self.element = list[0] #Ag
        self.feature = []
        if self.element == "Ag":
            self.feature = [1]
        else:
            self.feature = [0]
        self.x = float(list[1])
        self.y = float(list[2])
        self.z = float(list[3])
        self.position = [self.x,self.y,self.z]

    def __str__(self):
        return str(self.num)