from .basicextractor import BasicExtractor

class TweetContainsTagExtracor(BasicExtractor):
    def applyTransformation(self):
        self.data['Has_Tag'] = self.data['Tweet'].apply(lambda row: '@' in str(row))