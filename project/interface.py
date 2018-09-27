# encoding=utf-8
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pickle as pk
import os

class SupperModel:
    def __init__(self, model, param=None, **kags):
        """ Unified training and prediction process

        Parameters
        ----------
        model : obj
            The model of your training model, such as xgb/lgb.
        params : dict or None, optional (default=None)
            The params of your model.
        kags : dict or None
            Depending on your requirement.
        """
        self.model = model
        self.params = param

    def acc(self, Y, Y_pred):
        """ Showing some metrics about the training process

        Parameters
        ----------
        Y : list, numpy 1-D array, pandas.Series
            The ground truth on the val dataset.
        Y_pred : list, numpy 1-D array, pandas.Series
            The predict by your model on the val dataset.
        """
        Y = list(Y); Y_pred = list(Y_pred)
        print('precision:', precision_score(Y, Y_pred))
        print('accuracy:', accuracy_score(Y, Y_pred))
        print('recall:', recall_score(Y, Y_pred))
        print('micro_F1:', f1_score(Y, Y_pred, average='micro'))
        print('macro_F1:', f1_score(Y, Y_pred, average='macro'))

    def train(self, X_train, y_train):
        """ Training the model

        Parameters
        ----------
        X_train : pandas.DataFrame, numpy.array, list
            The train data except the label
        y_train : list, numpy 1-D array, pandas.Series
            The label of the train data
        """
        pass
    
    def predict(self, X_test):
        """ Using the model trained predict

        Parameters
        ----------
        X_test : pandas.DataFrame, numpy.array, list
            The test dataset
        """
        pass

    def saveModel(self, save_path):
        """ Save the model during training process

        Parameters
        ----------
        save_path : str
            the model's save_path
        """
        if not os.path.exists('/'.join(os.path.split(save_path)[:-1])):
            os.makedirs('/'.join(os.path.split(save_path)[:-1]))
        with open(save_path, 'wb') as fw:
            pk.dump(self.model, fw)
    
    def loadModel(self, save_path):
        """ Load the model from the save_path

        Parameters
        ----------
        save_path : str
            the model's save_path

        Returns
        -------
        model : obj or None(if save_path don't exists)
            the model saved in the save_path
        """
        if not os.path.exists(save_path):
            return
        with open(save_path, 'rb') as fr:
            return pk.load(fr)
