# pylint: skip-file
import csv
import datetime
import gzip
import json
import os
import pathlib
import random
import itertools
import requests

#
# import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from sentence_transformers import util
from typing import List

CROSS_ENCODERS = [
    "cross-encoder/stsb-distilroberta-base",
    "cross-encoder/stsb-roberta-large",
    "cross-encoder/stsb-bert-large",
    "bert-base-uncased",
]
RESULTS_PATH = pathlib.Path("../data/results")
DATA_PATH = pathlib.Path("../data/")
b = 0


def similarity_sent2vec(
    sentences: List[List[str]],
    real_scores: List[float],
    verbose=False,
):
    return None


def sent2vec_accuracy(x: List[List[str]], y: List[float], verbose):
    scores = similarity_sent2vec(x, y, verbose=verbose)
    err = np.array([abs(score - pred_score) for score, pred_score in zip(y, scores)])

    scores = np.array(scores)

    acc = 1 - np.mean(err)
    pearson = pearsonr(scores, y)[0]
    spearman = spearmanr(scores, y)[0]

    print(f"pearson: {pearson:2.3f}")
    print(f"spearman: {spearman:2.3f}")
    return f"acc: {acc}, p: {pearson}, s: {spearman}"


def compute_accuracy(ir, _y, y_hat):
    return 1 - compute_error(ir, _y, y_hat)


def compute_error(ir, _y, y_hat):
    return np.mean([abs(y - yh) for y, yh in zip(_y, y_hat)])


def compute_score_pairs(score):
    sc = np.diag(score)
    return sc.ravel()


def print_scores(ir, name, _x, _y):
    scores = np.diag(ir.predict_proba(_x))
    scores = compute_score_pairs(scores)
    acc = compute_accuracy(ir, scores, _y)
    pearson = pearsonr(scores, _y)[0]
    spearman = spearmanr(scores, _y)[0]

    print(name)
    print(f"acc: {acc:2.3f}")
    print(f"pearson: {pearson:2.3f}")
    print(f"spearman: {spearman:2.3f}")

    return f"acc: {acc}, p: {pearson}, s: {spearman}"


def print_predictions(ir, x_tst, y_tst):
    ctr = 0
    for sentences, score in zip(x_tst, y_tst):
        if ctr == 10:
            break
        y_hat = np.array(ir.predict_proba([sentences]))
        correct = int(abs(y_hat - score) < 0.3)
        print(f"{correct}, {sentences}, score={score}, pred_score = {y_hat[0]}")
        ctr += 1


def save_data_to_csv(data, models):
    date_time = datetime.datetime.now().replace(second=0, microsecond=0)
    file = "performance_results_{}.csv".format(str(date_time).replace(" ", "_"))

    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)
    path = RESULTS_PATH / file

    with open(path, "w") as f:
        writer = csv.writer(f)

        for test_dataset, test_perf in data.items():
            writer.writerow([test_dataset])
            writer.writerow(["train set"] + models)
            for train_dataset, perf in test_perf.items():
                row = [train_dataset]
                for model, acc in perf.items():
                    row.append(acc)
                writer.writerow(row)
            writer.writerow([])
            writer.writerow([])


def save_results_to_csv(results, models):
    date_time = datetime.datetime.now().replace(second=0, microsecond=0)
    file = str(date_time) + ".csv"
    path = RESULTS_PATH

    if not os.path.exists(path):
        os.mkdir(path)
    path = path / file

    with open(path, "w") as f:
        writer = csv.writer(f)

        row = ["dataset"]
        row.extend(models)
        writer.writerow(row)

        for dataset in results.keys():
            row = [dataset]
            for model in models:
                row.append(results[dataset][model])
            writer.writerow(row)


def save_to_csv(results, fieldnames, file, n_samples=""):
    date_time = datetime.datetime.now().replace(second=0, microsecond=0)
    file = "{}{}_{}.csv".format(file, n_samples, str(date_time).replace(" ", "_"))
    path = RESULTS_PATH

    if not os.path.exists(path):
        os.mkdir(path)
    path = path / file

    with open(path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in results:
            writer.writerow(data)


def get_data_faq(path, max_train_variations=None):
    X_trn, y_trn, X_tst, y_tst = [], [], [], []

    for doc in json.load(path.open("rt")):
        intent = doc["id"]

        y_trn.append(intent)
        X_trn.append(doc["question_set"][0])

        max_variations = max_train_variations or len(doc["variations"]) - 1
        for i in range(max_variations):
            y_trn.append(intent)
            X_trn.append(doc["variations"][i])

        for i in range(max_variations, len(doc["variations"])):
            X_tst.append(doc["variations"][i])
            y_tst.append(intent)
    return X_trn, y_trn, X_tst, y_tst


def get_datasets(train: List[str] = None, dev: List[str] = None, test: List[str] = None):
    _train = [train] if isinstance(train, str) else train
    _dev = [dev] if isinstance(dev, str) else dev
    _test = [test] if isinstance(test, str) else test

    _train, _dev, _test = _train or [], _dev or [], _test or []
    x_trn, y_trn, x_dev, y_dev, x_tst, y_tst = [], [], [], [], [], []
    for train_file in _train:
        x, y = get_data(train_file, "train")
        x_trn.extend(x)
        y_trn.extend(y)
    for dev_file in _dev:
        x, y = get_data(dev_file, "dev")
        x_dev.extend(x)
        y_dev.extend(y)
    for test_file in _test:
        x, y = get_data(test_file, "test")
        x_tst.extend(x)
        y_tst.extend(y)

    print(f"number of pairs - train: {len(y_trn)}, dev: {len(y_dev)}, test: {len(y_tst)}")
    return x_trn, y_trn, x_dev, y_dev, x_tst, y_tst


def get_data(filename, split):
    path = DATA_PATH / filename
    if "STS" in filename:
        x, y = get_data_sts(path)
    else:
        if filename == "stsbenchmark.tsv.gz" and not os.path.exists(path):  # TODO: add parameter of smth
            util.http_get("https://sbert.net/datasets/stsbenchmark.tsv.gz", str(path))
        x, y = get_data_csv(path, split)
    return x, y


def get_data_sts(path):
    x, y = [], []

    if os.path.exists(path):
        try:
            data = json.load(path.open("rt"))
            random.shuffle(data)
            for sample in data:
                sent1 = sample[0]
                sent2 = sample[1]
                score = sample[2]

                x.append([sent1, sent2])
                y.append(score)
        except json.decoder.JSONDecodeError:
            pass
    else:
        print("path doesn't exist {}".format(path))  # TODO: absolute path for DATA
    return x, y


def get_data_csv(path, split):
    x, y = [], []
    if os.path.exists(path):
        with gzip.open(path, "rt", encoding="utf8") as file:
            reader = csv.DictReader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if row["split"] == split:
                    score = float(row["score"]) / 5.0  # Normalize score to range 0 ... 1
                    texts = [
                        row["sentence1"].replace("’", "'").replace("‚", "'").replace('"', "'").encode("utf-8").decode(),
                        row["sentence2"].replace("’", "'").replace("‚", "'").replace('"', "'").encode("utf-8").decode(),
                    ]
                    x.append(texts)
                    y.append(score)
    return x, y


def plot_learning_curve(dataset_name, model_name, tr_sizes, tr_errors, tst_errors, tr_color="b", tst_color="r"):
    fig, ax = plt.subplots()

    # ax.set_yscale('log')
    ax.plot(tr_sizes, tr_errors, lw=2, c=tr_color, label="training error")
    ax.plot(tr_sizes, tst_errors, lw=2, c=tst_color, label="dev error")
    ax.set_xlabel("training set size")
    ax.set_ylabel("error")

    ax.legend(loc=0)
    ax.set_xlim(0, np.max(tr_sizes))
    ax.set_ylim(0, 1)
    ax.set_title(f"Learning curve for {model_name}, fine-tuned {dataset_name}")
    ax.grid(True)

    filename = f"learning_curve_{model_name}_{dataset_name}.png".replace("sentence-transformers/", "").replace(
        ".json", ""
    )
    path = RESULTS_PATH / "figs"

    if not os.path.exists(path):
        if not os.path.exists(RESULTS_PATH):
            os.mkdir(RESULTS_PATH)
        os.mkdir(path)

    plt.savefig(path / filename)
    plt.show()


def get_json(file):
    data = []
    path = DATA_PATH / file
    if os.path.exists(DATA_PATH / file):
        data = json.load(path.open("rt"))
    return data


def save_to_json(file, data):
    path = DATA_PATH / file
    with open(path, "w") as outfile:
        if len(data) > 0:
            json.dump(data, outfile, separators=(", \n", ":"))
