from .basicextractor import BasicExtractor

class TweetContainsTagExtracor(BasicExtractor):
    def apply(self, feature_name):
        self.data['Has_Tag'] = self.data['Tweet'].apply(lambda row: '@' in str(row))