from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


class Metrics:

    @staticmethod
    def tpr(t, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred>t)
        return cm[1,1]/(cm[1,1]+cm[1,0])

    @staticmethod
    def fpr(t, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred > t)
        return cm[0, 1] / (cm[0, 1] + cm[0, 0])

    def roc(self,y_true, y_score):
        return (np.array([self.fpr(prob, y_true, y_score) for prob in np.linspace(1, 0, 10)]),
                np.array([self.tpr(prob, y_true, y_score) for prob in np.linspace(1, 0, 10)]),
                np.array([prob for prob in np.linspace(1, 0, 10)]))

    def roc_auc(self, y_true, y_score):
        fp = [self.fpr(prob, y_true, y_score) for prob in np.linspace(1, 0, 1000)]
        tp = [self.tpr(prob, y_true, y_score) for prob in np.linspace(1, 0, 1000)]
        return sum([(fp[i + 1] - fp[i]) * (tp[i] + tp[i + 1]) / 2 for i in range(len(fp) - 1)])

    def plot(self,y_true, y_score, area):
        plt.plot([self.fpr(prob,y_true,y_score) for prob in np.linspace(1,0,1000)],
         [self.tpr(prob,y_true,y_score) for prob in np.linspace(1,0,1000)], label='ROC curve(area={})'.format(area))
        plt.plot([0,1],[0,1] )
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.show()