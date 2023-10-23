class GeeKnnClassifier:
    def __init__(self, k=1, max_duplicates=None):
        self.k = k
        self.max_duplicates = max_duplicates if max_duplicates is not None else 5

    @property
    def k_nearest(self):
        return self.k + self.max_duplicates
