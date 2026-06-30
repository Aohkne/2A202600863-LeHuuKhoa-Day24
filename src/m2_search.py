"""M2 Hybrid Search — copy from Day 18 for full implementation."""


class SearchResult:
    def __init__(self, text, score, metadata):
        self.text = text
        self.score = score
        self.metadata = metadata


class HybridSearch:
    def index(self, chunks):
        pass

    def search(self, query, top_k=20):
        return []
