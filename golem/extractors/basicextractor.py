import pandas as pd
import json

class BasicExtractor:
    def exportJson(self, fullPath):
        with open(path, 'w') as exportFile:
            exportFile.write(self.data)

    def loadJson(self, fullPath):
        with open(fullPath, 'r') as infile:
            jsonFile = json.load(infile)
        self.data = pd.DataFrame.from_records(jsonFile)

    def applyTransformation(self, transformator):
        self.data = transformator.apply(data)