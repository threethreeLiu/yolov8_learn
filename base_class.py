"""
@File    : base_class.py
@Date    : 2024-06-13
@Author  : LiuTianSheng
@Software : yolo-learn
"""
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass




