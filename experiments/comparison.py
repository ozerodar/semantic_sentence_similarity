import os
import time
import argparse
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers import util

from experiments.model import Model
from experiments.utils import (
    print_scores,
    print_predictions,
    save_results_to_csv,
    get_data,
    compute_error,
    plot_learning_curve,
    sent2vec_accuracy,
    get_datasets,
    save_to_csv,
    save_data_to_csv,
    compute_accuracy,
)

TUNED_MODELS = "../data/tuned"
DATASETS = {
    "-": "-",
    "STSb_train": "STS_train_2021_stsb.json",
    "STSb_dev": "STS_dev_2021_stsb.json",
    "STSb_test": "STS_test_2021_stsb.json",
    "STSb_train_err": "STS_train_2021_stsb_errors.json",
    "STSb_dev_err": "STS_dev_2021_stsb_errors.json",
    "STSb_test_err": "STS_test_2021_stsb_errors.json",
    "HP_train": "STS_train_2021_harry_potter.json",
    "HP_dev": "STS_dev_2021_harry_potter.json",
    "HP_test": "STS_test_2021_harry_potter.json",
    "hard_test": "STS_test_hard_sentences.json",
}


def get_acronym(split: Union[List[str], str]):
    return split if isinstance(split, str) else "+".join(data for data in split)


def get_acronyms(train, dev, test):
    return get_acronym(train), get_acronym(dev), get_acronym(test)


def get_data_filenames(data: Union[List[str], str]):
    if isinstance(data, list):
        return [get_data_filenames(subset) for subset in data]
    else:
        return DATASETS[data]


def get_dataset_filenames(train, dev, test):
    return get_data_filenames(train), get_data_filenames(dev), get_data_filenames(test)


def compare_performance(data, model_names, retrain=False, verbose=False):
    results = {}
    for train, devel, test in data:
        trn, dev, tst = get_acronyms(train, devel, test)

        results[f"trn: {trn}, dev: {dev}, tst: {tst}"] = {}
        print(f"trn: {trn}, dev: {dev}, tst: {tst}")

        train_files, dev_files, test_files = get_dataset_filenames(train, devel, test)
        X_trn, y_trn, X_dev, y_dev, X_tst, y_tst = get_datasets(train_files, dev_files, test_files)

        for model_name in model_names:
            if model_name != "sent2vec" and model_name != "elmo":
                s = "## model: {}".format(model_name)

                IR = Model(
                    model_name,
                    pretrained_model=f"{TUNED_MODELS}/{trn}/{model_name}",
                    retrain=retrain,
                    x_dev=X_dev,
                    y_dev=y_dev,
                )
                if verbose:
                    print_predictions(IR, X_tst, y_tst)

                IR.fit(X_trn, y_trn, retrain=retrain, epochs=5)
                if retrain:
                    print("---------- fitting -------------")
                    if verbose:
                        print_predictions(IR, X_tst, y_tst)

                acc = print_scores(IR, s, X_tst, y_tst)
                print("-----------------------")

            elif model_name == "sent2vec" and ip is not None and port is not None:
                print("## model: {}".format(model_name))
                if len(y_trn) != 0:
                    acc = -1
                else:
                    acc = sent2vec_accuracy(X_tst, y_tst, verbose=verbose)
                print(f"acc: {acc}")
                print("-----------------------")
            results[f"trn: {trn}, dev: {dev}, tst: {tst}"][model_name] = acc

    print(results)
    save_results_to_csv(results, model_names)


def compare_learning_curves(data, model_names):
    results = {}
    for train, devel, test in data:
        trn, dev, tst = get_acronyms(train, devel, test)
        results[f"{tst}"] = {}

        train_files, dev_files, test_files = get_dataset_filenames(train, devel, test)
        X_trn, y_trn, X_dev, y_dev, X_tst, y_tst = get_datasets(train_files, dev_files, test_files)

        tr_sizes = list(range(0, len(y_trn), 20))
        # print(tr_sizes)

        for model_name in model_names:
            tr_errors = np.zeros_like(tr_sizes, dtype=float)
            tst_errors = np.zeros_like(tr_sizes, dtype=float)

            print("## model: {}".format(model_name))

            for index, size in enumerate(tr_sizes):
                # TODO: if no saved_path don't save anything
                IR = Model(
                    model_name,
                    pretrained_model=f"{TUNED_MODELS}/{trn}/{model_name}",
                    retrain=True,
                    x_dev=x_dev,
                    y_dev=y_dev,
                )
                if size > 1:
                    IR.fit(X_trn[0:size], y_trn[0:size], retrain=True)

                tr_errors[index] = compute_error(IR, X_trn, y_trn)
                tst_errors[index] = compute_error(IR, X_tst, y_tst)

                print("#### error: {}".format(tst_errors[index]))

            plot_learning_curve(test, model_name, tr_sizes, tr_errors, tst_errors)


def compare_exec_time(data, model_names):
    results = []
    train, devel, test = data
    train_files, dev_files, test_files = get_dataset_filenames(train, devel, test)
    X_trn, y_trn, X_dev, y_dev, X_tst, y_tst = get_datasets(train_files, dev_files, test_files)

    for model_name in model_names:
        if model_name != "sent2vec":

            IR = Model(
                model_name,
                pretrained_model=f"{TUNED_MODELS}/{data}/{model_name}",
                retrain=True,
                x_dev=X_dev,
                y_dev=y_dev,
            )
            start_time = time.time()
            IR.fit(X_trn, y_trn, retrain=True, epochs=1)
            train_time = time.time() - start_time

            start_time = time.time()
            IR.predict_proba(X_tst)
            pred_time = time.time() - start_time

            embeddings = IR.model.encode([x[0] for x in X_tst], convert_to_tensor=True)

            start_time = time.time()
            query = IR.model.encode(X_tst[0][1], convert_to_tensor=True)
            for emb in embeddings:
                util.pytorch_cos_sim(emb, query).squeeze().tolist()
            prec_pred_time = time.time() - start_time

            print("--- train %s seconds ---" % train_time)
            print("--- precoputed prediction %s seconds ---" % prec_pred_time)
            print("--- prediction %s seconds ---" % pred_time)

            results.append(
                {
                    "model": model_name,
                    "training time": "{:2.3f}".format(train_time),
                    "prediction time": "{:2.3f}".format(pred_time),
                }
            )

    save_to_csv(
        results, fieldnames=["model", "training time", "prediction time"], file="exec_time.csv", n_samples=len(y_tst)
    )


def compare(
    model_names,
    data,
    exec_time_model_names,
    exec_time_data,
    update_dataset=False,
    comp_performance=True,
    comp_exec_time=True,
    calc_learning_curve=True,
    verbose=False,
    retrain=True,
):
    if update_dataset:
        from experiments.datasetcreator import DatasetCreator

        stsb_dataset = DatasetCreator(
            train_file="stsbenchmark.tsv.gz",
            dev_file="stsbenchmark.tsv.gz",
            test_file="stsbenchmark.tsv.gz",
            dataset="stsb",
        )
        stsb_dataset.create(update=False, max_samples=100000)

        harry_potter_dataset = DatasetCreator(num_samples=500)
        harry_potter_dataset.create(update=True)

    if comp_performance:
        compare_performance(data, model_names, retrain=retrain, verbose=verbose)
    if calc_learning_curve:
        compare_learning_curves(data, model_names)
    if comp_exec_time:
        compare_exec_time(exec_time_data[0], exec_time_model_names)


if __name__ == "__main__":
    # model_names = ["sent2vec", "sentence-transformers/all-roberta-large-v1"]
    model_names = ["sentence-transformers/all-roberta-large-v1"]
    exec_time_model_names = ["sentence-transformers/all-roberta-large-v1"]

    data = [
        ("-", "-", "STSb_test"),
        ("STSb_train", "STSb_dev", "STSb_test"),
        ("-", "-", "STSb_test_err"),
        ("STSb_train", "STSb_dev", "STSb_test_err"),
        (["STSb_train", "STSb_train_err"], ["STSb_dev", "STSb_dev_err"], "STSb_test_err"),
        ("-", "-", "HP_test"),
        ("HP_train", "HP_dev", "HP_test"),
        ("-", "-", "hard_test"),
    ]

    exec_time_data = (("HP_train", "HP_dev", "HP_test"),)

    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("-p", action="store_true", help="Compare performance", required=False)
    parser.add_argument("-u", action="store_true", help="Update dataset", required=False, default=False)
    parser.add_argument("-t", action="store_true", help="Compare execution time", required=False, default=False)
    parser.add_argument("-l", action="store_true", help="Calculate learning curve", required=False, default=False)
    parser.add_argument("-r", action="store_true", help="Retrain", required=False, default=False)
    parser.add_argument("-v", action="store_true", help="Verbose", required=False, default=False)
    args = vars(parser.parse_args())

    comp_performance, update_dataset, comp_exec_time, calc_learning_curve, retrain, verbose = args.values()

    if "sent2vec" in model_names:
        print('sorry, sent2vec is not available')
    compare(
        model_names,
        data,
        exec_time_model_names,
        exec_time_data,
        update_dataset=update_dataset,
        comp_performance=comp_performance,
        comp_exec_time=comp_exec_time,
        calc_learning_curve=calc_learning_curve,
        retrain=retrain,
        verbose=verbose,
    )
