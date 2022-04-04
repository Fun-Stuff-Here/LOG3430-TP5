import json
from main import evaluate
from main import DEFAULT_TRAINING_SET
import unittest


class TestMain(unittest.TestCase):
	def setUp(self):
		self.baseline = evaluate(test_set_json_path=DEFAULT_TRAINING_SET)

	def f1_comparison(self, json_path: str):
		score_f1 = evaluate(test_set_json_path=json_path)
		self.assertAlmostEqual(self.baseline, score_f1, 2, "F1-score is not equivalent", 0.03)

	def test_nettoyage_train(self):
		pass

	def test_nettoyage_test(self):
		pass

	def test_permutation_train(self):
		pass

	def test_permutation_test(self):
		pass

	def test_triple_train(self):
		pass

	def test_triple_test(self):
		pass

	def test_duplicate_train(self):
		pass

	def test_duplicate_test(self):
		pass

