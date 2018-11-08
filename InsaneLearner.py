import LinRegLearner as lrl
import BagLearner as bl
import numpy as np
class InsaneLearner(object):
    def __init__(self,learner= lrl.LinRegLearner,verbose = False):
        self.learners=[]
        for i in range(0, 20):
            self.learners.append(bl.BagLearner(learner,{},20,False,False))
    def author(self):
        return 'rjayakrishnan3'  # replace tb34 with your Georgia Tech username
    def addEvidence(self, dataX, dataY):
        for i in range(0, 20):
           self.learners[i].addEvidence(dataX,dataY)
    def query(self, points):
        allpredictions=np.empty(shape=(20,len(points)))
        for i in range (20):
            allpredictions[i,:]=self.learners[i].query(points)
        predictions=np.mean(allpredictions,axis=0)
        return predictions

