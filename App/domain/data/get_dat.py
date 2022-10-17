#class pour recuperer le dataset 

from this import d
from typing import Dict, Tuple
from numpy import ndarray
import pandas as pd 
import matplotlib as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from yaml import load


class get_datairis():
    def __init__(self, test_size: float = 0.33,  random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        
    
    @staticmethod
    def _get_iris() -> Dict:
        return datasets.load_iris(return_X_y=True)   
    
    
    @staticmethod
    def _split_data(iris_dictionary: Dict) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        return train_test_split(*iris_dictionary)
    
    
    def get_data(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        return self._split_data(self.get_iris())

