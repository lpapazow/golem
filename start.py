from golem.extractors import basicextractor
from golem.parsers import csvtojson
from golem.extractors import tweetcontainstagextractor

def main():
    parser = csvtojson.CsvToJsonParser('data/data_csv/2018-E-c-En-train.txt')
    jsonData = parser.exportJson('data/data_json/data.json')

    sampleExtractor = tweetcontainstagextractor.TweetContainsTagExtracor()
    sampleExtractor.loadJson('data/data_json/data.json')
    sampleExtractor.apply(sampleExtractor)
    jsonData = parser.exportJson('data/data_json/data_contains_tag.json')

if __name__ == '__main__':
    main()
