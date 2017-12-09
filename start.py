import os
from golem.extractors import basicextractor
from golem.parsers import csvtojson
from golem.extractors import tweetcontainstagextractor

def main():
    parser = csvtojson.CsvToJsonParser('data/data_csv/2018-E-c-En-train.txt')

    if not os.path.exists('data/data_json'):
        os.makedirs('data/data_json')

    jsonData = parser.exportJson('data/data_json/data.json')
    sampleExtractor = tweetcontainstagextractor.TweetContainsTagExtracor()
    sampleExtractor.loadJson('data/data_json/data.json')
    sampleExtractor.applyTransformation()
    jsonData = parser.exportJson('data/data_json/data_contains_tag.json')

if __name__ == '__main__':
    main()
