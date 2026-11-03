import json
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import build_opener, ProxyHandler

from recommender.engine import Recommender
from recommender.server import create_server


DATA = Path(__file__).resolve().parents[1] / "data" / "interactions.csv"


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = Recommender(DATA)

    def test_personalized_excludes_seen(self):
        results = self.engine.recommend("u1", 3)
        self.assertEqual([r["item_id"] for r in results], ["i4", "i3", "i5"])
        self.assertEqual(results[0]["source"], "collaborative")
        self.assertEqual(results[-1]["source"], "popular")

    def test_cold_start(self):
        results = self.engine.recommend("new-user", 2)
        self.assertEqual([r["item_id"] for r in results], ["i2", "i1"])
        self.assertTrue(all(r["source"] == "popular" for r in results))

    def test_invalid_limit(self):
        for limit in (0, 101):
            with self.assertRaises(ValueError):
                self.engine.recommend("u1", limit)


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(Recommender(DATA), port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.client = build_opener(ProxyHandler({}))

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path):
        return self.client.open(f"http://127.0.0.1:{self.server.server_port}{path}", timeout=5)

    def test_health_and_recommendations(self):
        with self.request("/health") as response:
            self.assertEqual(json.load(response), {"status": "ok", "items": 5})
        with self.request("/recommendations?user_id=u1&limit=2") as response:
            body = json.load(response)
            self.assertEqual(body["user_id"], "u1")
            self.assertEqual([r["item_id"] for r in body["recommendations"]], ["i4", "i3"])

    def test_bad_requests(self):
        for path, status in [("/missing", 404), ("/recommendations", 400), ("/recommendations?user_id=u1&limit=no", 400), ("/recommendations?user_id=u1&limit=101", 400)]:
            with self.assertRaises(HTTPError) as caught:
                self.request(path)
            self.assertEqual(caught.exception.code, status)
            caught.exception.close()
