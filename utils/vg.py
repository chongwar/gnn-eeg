# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AUTHOR: Delia Fano Yela
# DATE:  December 2018
# CONTACT: d.fanoyela@qmul.ac.uk
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ------------------------------------------------------------------------------
# NODE CLASS
# ------------------------------------------------------------------------------
class Node:
    # --------------------------------------------------------------------------
    def __init__(self, value=None, data=None, left=None, right=None):
        self.value  = value     # Float: Value of the node
        self.data   = data      # Float: Data value associated with that node
        self.left   = left      # Node: Left child
        self.right  = right     # Node: Right child
    # --------------------------------------------------------------------------
    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return self.value == other.value and self.data == other.data
        else:
            return False
    # ------------------------------------------------------------------------------
    def __ne__(self, other):
        """Override the default Unequal behavior"""
        return self.value != other.value or self.data != other.data
    # ------------------------------------------------------------------------------
    def add(self, node):
        if  not self.value and self.value != 0:
            self.value = node.value
            self.data = node.data
        else:
            if node.value < self.value:
                if not self.left and self.left != 0 : self.left = Node()
                self.left.add(node)
            else:
                if not self.right and self.right != 0 : self.right = Node()
                self.right.add(node)
    # --------------------------------------------------------------------------
    def getKids(self):
        if not self.left and not self.right:
            return None
        kids = {}
        if self.left:
            kids["left"] =  self.left
        if self.right:
            kids["right"] = self.right
        return kids

    # --------------------------------------------------------------------------
    def delKid(self, keyword):
        if keyword == 'left': self.left = None
        if keyword == 'right': self.right = None

    # --------------------------------------------------------------------------
    def getLeftBranch(self):
        if self.left:
            return [self.left] + self.left.getLeftBranch()
        else:
            return []

    # --------------------------------------------------------------------------
    def getRightBranch(self):
        if self.right:
            return [self.right] + self.right.getRightBranch()
        else:
            return []

    # --------------------------------------------------------------------------
    def getLeftovers(self, check = None):
        if check == None: check = []

        lout = []
        if self.right :
            lout += self.right.getRightTree()

            if self.right.left:
                lout += self.right.left.getLeftRightTree()

        if self.left  :
            lout += self.left.getLeftTree()

            if self.left.right:
                lout += self.left.right.getRightLeftTree()


        if len(lout)>0 : check.append([self, lout])

        if self.right:
            self.right.getLeftovers(check)
        if self.left:
            self.left.getLeftovers(check)

        return check

    # --------------------------------------------------------------------------
    def getLeftRightTree(self):
        if self.right and self.left:
            return self.right.root2list() + self.left.getLeftRightTree()
        elif self.right:
            return self.right.root2list()
        elif self.left:
            return self.left.getLeftRightTree()
        else:
            return []
    # --------------------------------------------------------------------------
    def getRightLeftTree(self):
        if self.left and self.right:
            return self.left.root2list() + self.right.getRightLeftTree()
        elif self.left:
            return self.left.root2list()
        elif self.right:
            return self.right.getRightLeftTree()
        else:
            return []

    # --------------------------------------------------------------------------
    def getRightTree(self):
        if self.right:
            return self.right.root2list()
        else:
            return []

    # --------------------------------------------------------------------------
    def getLeftTree(self):
        if self.left:
            return self.left.root2list()
        else:
            return []

    # --------------------------------------------------------------------------
    def root2list(self, out = None):
        if out == None:
            out = []

        out.append(self)

        if self.left: self.left.root2list(out)
        if self.right: self.right.root2list(out)

        return out

    # ------------------------------------------------------------------------------
    def visibleNodes(self,connect = None):
        if connect == None:
            connect = []

        visible = []
        if self.right:
            visible += [self.right.value]

            if self.right.left:
                visible += [x.value for x in self.right.getLeftBranch()]

        if self.left:
            visible += [self.left.value]

            if self.left.right:
                visible += [x.value for x in self.left.getRightBranch()]

        if len(visible)>0 : connect.append([self.value, visible])


        if self.right:
            self.right.visibleNodes(connect)
        if self.left:
            self.left.visibleNodes(connect)

        return connect

    # ------------------------------------------------------------------------------
    def continuityRight(self):
        if self.right and (self.right.data == self.data):
            return self.right.continuityRight()
        else:
            return self.value

    # ------------------------------------------------------------------------------
    def continuityLeft(self):
        if self.left and (self.left.data == self.data):
            return self.left.continuityLeft()
        else:
            return self.value

    # ------------------------------------------------------------------------------
    def horizontalVisibleNodes(self,connect = None):
        if connect == None:
            connect = []

        visible = []

        if self.right:
            visible += [self.right.continuityLeft()]

            if self.right.left:
                for x in self.right.getLeftBranch():
                    if (not x.left) or (x.left and x.left.data != x.data):
                        visible += [x.value]

        if self.left:
            visible += [self.left.continuityRight()]

            if self.left.right:
                for x in self.left.getRightBranch():

                    if (not x.right) or (x.right and x.right.data != x.data):
                        visible += [x.value]

        if len(visible)>0 : connect.append([self.value, visible])

        if self.right:
            self.right.horizontalVisibleNodes(connect)
        if self.left:
            self.left.horizontalVisibleNodes(connect)

        return connect


    # --------------------------------------------------------------------------
    def display(self):
        if self.value >= 0:

            if self.left:
                left = self.left.value
            else:
                left = None

            if self.right:
                right = self.right.value
            else:
                right = None

            print('----------------------------------------------------------------')
            print('VALUE : ', self.value, 'DATA : ', self.data, ';    LEFT :', left, ';     RIGHT : ', right,';')

            if left: self.left.display()
            if right: self.right.display()

# ------------------------------------------------------------------------------
# METHODS
# ------------------------------------------------------------------------------
def build(series, delay = None, timeLine = None):
    root = Node()

    if not (type(series) == list): series = series.tolist()

    # For natural visibility graphs it needs to be smallest index first in case of equal amplitude
    sorted_index = sorted(range(len(series)), reverse=True, key=lambda k: series[k])

    data_values  = [series[i] for i in sorted_index]

    if timeLine: sorted_index = [timeLine[idx] for idx in sorted_index]
    elif delay: sorted_index = [i + delay for i in sorted_index]

    for (index, data_value) in zip(sorted_index,data_values):
        root.add(Node(value = index, data = data_value))

    return root


# ------------------------------------------------------------------------------
def sign(x):
    if x > 0:
        return 1.
    elif x < 0:
        return -1.
    elif x == 0:
        return 0.
    else:
        return x


# ------------------------------------------------------------------------------
def isEqual(root01, root02):
    if root01 == root02:
        return True
        if (root01.left and root02.left):
            isEqual(root01.left, root01.left)
        else:
            return False
        if (root01.right and root02.right):
            isEqual(root01.right, root01.right)
        else:
            return False

    else:
        return False


# ------------------------------------------------------------------------------
def visibility(series, timeLine = None, type = "horizontal"):
    if timeLine == None: timeLine = range(len(series))

    root = build(series, timeLine = timeLine)

    if type == 'horizontal':
        return root.horizontalVisibleNodes()

    elif type == 'natural':
        allNatural = []
        checkList = root.getLeftovers()

        for element in checkList:
            naturalVisible = []
            j = element[0].value

            for node in element[1]:
                i = node.value

                if i != j :

                    if i < j:
                        tb = j
                        yb = float(element[0].data)
                        ta = i
                        ya = float(node.data)
                    else:
                        tb = i
                        yb = float(node.data)
                        ta = j
                        ya = float(element[0].data)

                    a = timeLine.index(ta)
                    b = timeLine.index(tb)

                    yc = series[a+1:b]
                    tc = timeLine[a+1:b]

                    if all( yc[k] < (ya + (yb - ya)*(tc[k] - ta)/(tb-ta)) for k in range(len(yc)) ):
                        naturalVisible.append(node.value)

            if len(naturalVisible)>0 : allNatural.append([element[0].value, naturalVisible])

        return root.visibleNodes() + allNatural


# ------------------------------------------------------------------------------
def merge(list_roots_in): #[node01, node02]

    if len(list_roots_in) == 0:
        return None
    elif len(list_roots_in) == 1:
        return list_roots_in[0]
    else:
        # sort in ascending order by the node values in list:
        vroots = [x.value for x in list_roots_in]
        sidx = sorted(range(len(vroots)), reverse=False, key=lambda k: vroots[k])
        list_roots = [list_roots_in[i] for i in sidx]
        # find the node with the maximum data point in the list:
        # index --> Find index of first " " in mylist
        root = list_roots[[x.data for x in list_roots].index(max([x.data for x in list_roots]))]
        # Get the children of the maximum root
        root_kids = root.getKids()
        # Set the threshold :
        th =  root.value
        # initialise the pool of nodes to process:
        iter_pool = list_roots[:]
        iter_pool.remove(root)
        pool = iter_pool[:]
        # add the kids from the leftovers that are on the opposite side of the threshold to their parent
        for node in iter_pool:
            kids = node.getKids()
            if kids:
                for k in kids.items():
                    if sign(node.value - th) != sign(k[1].value - th):
                        pool.append(k[1])
                        node.delKid(k[0])

        # add the children of the maximum node
        if bool(root_kids):
            for k in root_kids.items():
                pool.append(k[1])
                root.delKid(k[0])

        # divide the pool according to the threshold
        smaller = [ x for x in pool if x.value < th]
        bigger  = [ x for x in pool if x.value > th]

        # return the node recursively:
        return(Node(value = root.value, data = root.data, left = merge(smaller), right = merge(bigger)))
    

if __name__ == '__main__':
    pass
