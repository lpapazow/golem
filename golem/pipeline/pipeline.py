from ..transformers.basic_transformer import BasicTransformer
import json
import pandas as pd


class Pipeline:
    def __init__(self, *transformers):
        self.transformers = transformers

    def remove_transformers(self, *transformers):
        for transformer in self.transformers:
            if transformer in transformers:
                self.transformers.remove(transformer) 

    def transform(self, data):
        for transformer in self.transformers:
            data = transformer.transform(data)
        return data