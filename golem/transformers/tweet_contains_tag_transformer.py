from .basic_transformer import BasicTransformer


class TweetContainsTagTransformer(BasicTransformer):
    def transform(self, data):
        data['Has_Tag'] = data['Tweet'].apply(lambda row: '@' in str(row))
        return data