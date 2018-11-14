import math
class ClusterNode:
    closeEnough # distance to be within to be considered similar
    numericAttributeIndices # indicies within data that hold numeric values

    def __init__(self,data,closeEnough,numericAttributeIndices = []):
        self.data = []
        self.data = data
        ClusterNode.closeEnough = closeEnough
        ClusterNode.numericAttributeIndices = numericAttributeIndices

    @staticmethod
    def similarityScore(node1,node2):
        """Compare two data notes and return a similarity score [0-1]."""
        similarity_score = 0.0
        numericFlag = True if len(ClusterNode.numericAttributeIndices) > 0 else False
        node1Numeric = []
        node2Numeric = []
        for i  in len(node1.data):
            if numericFlag and i in ClusterNode.numericAttributeIndices:
                node1Numeric.append(node1.data[i])
                node2Numeric.append(node2.data[i])
            else if node1.data[i] == node2.data[i]:
                    similarity_score += 1.0
        if distance(node1Numeric,node2Numeric) < ClusterNode.closeEnough:
            similarity_score += 1.0

        return featureScale(similarity_score,len(node1.data) - len(ClusterNode.numericAttributeIndices))

    @staticmethod
    """ Return normailized value between [0-1] given range """
    def featureScale(val,max,min=0.0):
        return val - min /  max - min

    @staticmethod
    def distance(node1Attributes,node2Attributes):
    """return distance between two points given lists of numeric attributes"""
        return math.sqrt([sum(i) for i in zip(node1Attributes,node2Attributes)])

    @staticmethod
    def setCloseEnough(closeEnough):
        ClusterNode.closeEnough = closeEnough
    @staticmethod
    def getCloseEnough():
        return ClusterNode.closeEnough

if __name__ == '__main__':
    print('Node for clustering')
