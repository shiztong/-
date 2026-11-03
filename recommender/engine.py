import csv
import math
from collections import defaultdict
from pathlib import Path


class Recommender:
    """User-based cosine collaborative filtering with popularity fallback."""

    def __init__(self, data_path):
        self.users = defaultdict(dict)
        self.items = {}
        self.popularity = defaultdict(float)
        with Path(data_path).open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                rating = float(row["rating"])
                if not math.isfinite(rating) or not 0 < rating <= 5:
                    raise ValueError("rating must be in (0, 5]")
                self.users[row["user_id"]][row["item_id"]] = rating
                self.items[row["item_id"]] = row["title"]
        for ratings in self.users.values():
            for item, rating in ratings.items():
                self.popularity[item] += rating

    def recommend(self, user_id, limit=10):
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        seen = self.users.get(user_id, {})
        scores = defaultdict(float)
        weights = defaultdict(float)
        norm = math.sqrt(sum(v * v for v in seen.values()))
        if norm:
            for other_id, ratings in self.users.items():
                if other_id == user_id:
                    continue
                dot = sum(value * ratings.get(item, 0) for item, value in seen.items())
                other_norm = math.sqrt(sum(v * v for v in ratings.values()))
                similarity = dot / (norm * other_norm)
                if similarity <= 0:
                    continue
                for item, rating in ratings.items():
                    if item not in seen:
                        scores[item] += similarity * rating
                        weights[item] += similarity
        ranked = sorted(scores, key=lambda i: (-scores[i] / weights[i], -self.popularity[i], i))
        results = [self._result(i, scores[i] / weights[i], "collaborative") for i in ranked]
        fallback = sorted(self.items, key=lambda i: (-self.popularity[i], i))
        results.extend(self._result(i, self.popularity[i], "popular") for i in fallback if i not in seen and i not in scores)
        return results[:limit]

    def _result(self, item, score, source):
        return {"item_id": item, "title": self.items[item], "score": round(score, 6), "source": source}
