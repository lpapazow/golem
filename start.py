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
    data = parser.get_pandas_data_frame()

    contains_tag_transformer = tweet_contains_tag_transformer.TweetContainsTagTransformer()

    sample_pipeline = pipeline.Pipeline(contains_tag_transformer)

    transformed_data = sample_pipeline.transform(data)

    parser.set_pandas_data_frame(transformed_data)
    parser.export_json('data/data_json/data_contains_tag.json')

if __name__ == '__main__':
    main()
