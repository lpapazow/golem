from ..transformers.basic_transformer import BasicTransformer
import json
import pandas as pd


class Pipeline:
    def __init__(self, data=None):
        self.data = data

    def load_json(self, fullPath):
        with open(fullPath, 'r') as infile:
            jsonFile = json.load(infile)
        self.data = pd.DataFrame.from_records(jsonFile)

    def transform(self, *transformers):
        for transformer in transformers:
            self.data = transformer.transform(self.data)
        return self.data