import os
from golem.transformers import basic_transformer
from golem.transformers import tweet_contains_tag_transformer
from golem.parsers import csvtojson
from golem.pipeline import pipeline

def main():
    parser = csvtojson.CsvToJsonParser('data/data_csv/2018-E-c-En-train.txt')

    if not os.path.exists('data/data_json'):
        os.makedirs('data/data_json')

    parser.export_json('data/data_json/data.json')
    sample_pipeline = pipeline.Pipeline()
    sample_pipeline.load_json('data/data_json/data.json')

    contains_tag_transformer = tweet_contains_tag_transformer.TweetContainsTagTransformer()

    data = sample_pipeline.transform(contains_tag_transformer)

    parser.set_pandas_data_frame(data)
    parser.export_json('data/data_json/data_contains_tag.json')

if __name__ == '__main__':
    main()
