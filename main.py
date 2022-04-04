import json
from vocabulary_creator import VocabularyCreator
from renege import RENEGE
from email_analyzer import EmailAnalyzer

DEFAULT_TRAINING_SET= "test_set.json"

def evaluate(test_set_json_path: str):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total = 0
    analyzer = EmailAnalyzer()
    with open(test_set_json_path) as email_file:
        new_emails = json.load(email_file)

    i = 0
    email_count = len(new_emails["dataset"])

    print("Evaluating emails ")
    for e_mail in new_emails["dataset"]:
        i += 1
        print("\rEmail " + str(i) + "/" + str(email_count), end="")

        new_email = e_mail["mail"]
        subject = new_email["Subject"]
        body = new_email["Body"]
        spam = new_email["Spam"]

        if (analyzer.is_spam(subject, body)) and (spam == "true"):
            tp += 1
        if (not (analyzer.is_spam(subject, body))) and (spam == "false"):
            tn += 1
        if (analyzer.is_spam(subject, body)) and (spam == "false"):
            fp += 1
        if (not (analyzer.is_spam(subject, body))) and (spam == "true"):
            fn += 1
        total += 1
    
    print("")
    print("\nAccuracy: ", round((tp + tn) / (tp + tn + fp + fn), 2))
    precision = tp / (tp + fp)
    print("Precision: ", round(precision, 2))
    recall = tp / (tp + fn)
    print("Recall: ", round(recall, 2))
    f1 = 2*(precision * recall)/(precision+recall)
    print(f"F1-score {f1}")
    return f1


if __name__ == "__main__":

    # 1. Creation de vocabulaire.
    vocab = VocabularyCreator()
    vocab.create_vocab()

    # 2. Classification des emails et initialisation des utilisateurs et des groupes.
    renege = RENEGE()
    renege.classify_emails()

    #3. Evaluation de performance du modele avec la fonction evaluate()
    f1 = evaluate(test_set_json_path=DEFAULT_TRAINING_SET)
