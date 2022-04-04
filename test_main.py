import json
import random

from main import evaluate
from main import DEFAULT_TRAINING_SET
import unittest
from text_cleaner import TextCleaning


class TestMain(unittest.TestCase):
	def setUp(self):
		self.baseline = evaluate(test_set_json_path=DEFAULT_TRAINING_SET)
		self.cleaner = TextCleaning()

	def f1_comparison(self, json_path: str):
		score_f1 = evaluate(test_set_json_path=json_path)
		self.assertAlmostEqual(self.baseline, score_f1, delta=0.03)

	def test_nettoyage_train(self):
		path = "json-metamorphique/train_clean.json"
		with open("train_set.json") as file:
			dataset = json.load(file)
			for row in dataset["dataset"]:
				row["mail"]["Subject"] = " ".join(self.cleaner.clean_text(row["mail"]["Subject"]))
				row["mail"]["Body"] = " ".join(self.cleaner.clean_text(row["mail"]["Body"]))
			out_file = open(path, "w")
			json.dump(dataset, out_file, indent=6)
			out_file.close()
		self.f1_comparison(path)

	def test_nettoyage_test(self):
		path = "json-metamorphique/test_clean.json"
		with open("test_set.json") as file:
			dataset = json.load(file)
			for row in dataset["dataset"]:
				row["mail"]["Subject"] = " ".join(self.cleaner.clean_text(row["mail"]["Subject"]))
				row["mail"]["Body"] = " ".join(self.cleaner.clean_text(row["mail"]["Body"]))
			out_file = open(path, "w")
			json.dump(dataset, out_file, indent=6)
			out_file.close()
		self.f1_comparison(path)

	def test_permutation_train(self):
		path = "json-metamorphique/train_shuffle.json"
		with open("train_set.json") as file:
			dataset = json.load(file)
			for row in dataset["dataset"]:
				tokens = self.cleaner.clean_text(row["mail"]["Body"])
				if len(tokens) < 3: continue
				index1, index2 = random.randint(0, len(tokens) - 1), random.randint(0, len(tokens) - 1)
				tokens[index1], tokens[index2] = tokens[index2], tokens[index1]
				row["mail"]["Body"] = " ".join(tokens)
			out_file = open(path, "w")
			json.dump(dataset, out_file, indent=6)
			out_file.close()
		self.f1_comparison(path)

	def test_permutation_test(self):
		path = "json-metamorphique/test_shuffle.json"
		with open("test_set.json") as file:
			dataset = json.load(file)
			for row in dataset["dataset"]:
				tokens = self.cleaner.clean_text(row["mail"]["Body"])
				if len(tokens) < 3: continue
				index1, index2 = random.randint(0, len(tokens)-1), random.randint(0, len(tokens)-1)
				tokens[index1], tokens[index2] = tokens[index2], tokens[index1]
				row["mail"]["Body"] = " ".join(tokens)
			out_file = open(path, "w")
			json.dump(dataset, out_file, indent=6)
			out_file.close()
		self.f1_comparison(path)

	def test_triple_train(self):
		path = "json-metamorphique/train700x3.json"
		with open("train_set.json") as file:
			dataset = json.load(file)
			new_dataset = {}
			new_dataset["dataset"] = []

			for row in dataset["dataset"]:
				new_dataset["dataset"].append(row)
				new_dataset["dataset"].append(row)
				new_dataset["dataset"].append(row)
			out_file = open(path, "w")
			json.dump(new_dataset, out_file, indent=6)
			out_file.close()
		self.f1_comparison(path)

	def test_triple_test(self):
		path = "json-metamorphique/train300x3.json"
		with open("test_set.json") as file:
			dataset = json.load(file)
			new_dataset = {}
			new_dataset["dataset"] = []

			for row in dataset["dataset"]:
				new_dataset["dataset"].append(row)
				new_dataset["dataset"].append(row)
				new_dataset["dataset"].append(row)
			out_file = open(path, "w")
			json.dump(new_dataset, out_file, indent=6)
			out_file.close()
		self.f1_comparison(path)

	def test_duplicate_train(self):
		path = "json-metamorphique/train_words.json"
		with open("train_set.json") as file:
			dataset = json.load(file)
			for row in dataset["dataset"]:
				row["mail"]["Body"] += row["mail"]["Body"]
			out_file = open(path, "w")
			json.dump(dataset, out_file, indent=6)
			out_file.close()
		self.f1_comparison(path)

	def test_duplicate_test(self):
		path = "json-metamorphique/test_words.json"
		with open("test_set.json") as file:
			dataset = json.load(file)
			for row in dataset["dataset"]:
				row["mail"]["Body"] += row["mail"]["Body"]
			out_file = open(path, "w")
			json.dump(dataset, out_file, indent=6)
			out_file.close()
		self.f1_comparison(path)
