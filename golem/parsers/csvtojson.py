import pandas as pd
import json

class CsvToJsonParser:

    INPUT_SEPARATOR = "\t"
    INPUT_HEADER_INDEX = 0

    def __init__(self, path_to_csv, separator=INPUT_SEPARATOR, header=INPUT_HEADER_INDEX):
        self.data = pd.read_csv(path_to_csv, sep=separator, header=header)

    def exportJson(self, path):
        data_json = self.data.to_json()
        with open(path, 'w') as exportFile:
            exportFile.write(data_json)

    def getPandasDataFrame(self):
        return self.data