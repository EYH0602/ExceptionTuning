import os
import sys
import re
import random
import json
from tqdm import tqdm
import time
import functools
import signal
from transformers import RobertaTokenizer
import matplotlib.pyplot as plt
import fire
import logging
from os.path import join as pjoin, abspath
from multiprocessing import Pool


def read_err_file(path):
    input = open(path, "r", encoding="utf-8").read()
    return input


def find_error_type(text):
    # todo: fix None
    lines = text.splitlines()

    # Extract the string before ":" in the fourth line
    if lines:
        last_line = lines[-1].strip()
        index_colon = last_line.find(":")

        if index_colon != -1:
            extracted_string = last_line[:index_colon].strip()
            return extracted_string
    else:
        return "NoError"


def find_error_line(text):
    lines = text.splitlines()
    if len(lines) > 1:
        second_line_from_bottom = lines[-2].strip()
        return second_line_from_bottom
    else:
        return ""


def get_err(file_path: str) -> tuple[str, str]:
    with open(file_path) as f:
        traceback = f.read()
    error_type = find_error_type(traceback)
    error_line = find_error_line(traceback)
    return str(error_type), error_line


def process_one_solution(
    fuzz_dir_path, code_dir_path, p: str, s: str, cont: int
) -> tuple[list[str], dict]:
    fuzz_path = pjoin(fuzz_dir_path, p, s, "default")
    if not os.path.exists(fuzz_path):
        return [], {}
    err_path = pjoin(fuzz_path, "stderr")
    errs = [get_err(pjoin(err_path, f)) for f in sorted(os.listdir(err_path))]
    js = {}

    # use first instance as data for faster training
    js["index"] = str(cont)
    js["label"] = errs[0][0]
    js["error"] = errs[0][1]
    with open(pjoin(code_dir_path, p, s), encoding="latin-1") as f:
        js["code"] = f.read()
    return list(map(lambda x: x[0], errs)), js


def main(
    fuzz_dir_path="fuzz",
    code_dir_path="Project_CodeNet_Python800",
    jobs=60,
):
    # initialize for two plots
    err_dict: dict[str, list[str]] = {}
    err_type_dict: dict[str, int] = {}

    f_train = open("train_err_cls.jsonl", "w")
    f_valid = open("valid_err_cls.jsonl", "w")
    f_test = open("test_err_cls.jsonl", "w")

    cont = 0

    logging.info("Collecting data")
    solutions = []
    for p in tqdm(os.listdir(code_dir_path)):
        full_path = os.path.join(code_dir_path, p)
        for s in os.listdir(full_path):
            solutions.append((fuzz_dir_path, code_dir_path, p, s, cont))
            cont += 1

    logging.info("Analyzing errors")
    with Pool(jobs) as pool:
        results = tqdm(
            pool.starmap(process_one_solution, solutions), total=len(solutions)
        )

    logging.info("Writing results to files")
    for p, res in zip(solutions, results):
        _, _, p, s, _ = p
        l, js = res
        err_dict[f"{p}/{s}"] = l

        for err_type in l:
            if err_type not in err_type_dict:
                err_type_dict[err_type] = 1
            else:
                err_type_dict[err_type] += 1

        r = random.random()
        if r < 0.6:
            split = "train"
        elif r < 0.8:
            split = "valid"
        else:
            split = "test"

        if split == "train":
            f_train.write(json.dumps(js) + "\n")
        elif split == "valid":
            f_valid.write(json.dumps(js) + "\n")
        elif split == "test":
            f_test.write(json.dumps(js) + "\n")
        else:
            raise NotImplementedError

    f_train.close()
    f_valid.close()
    f_test.close()

    # this is the dict for the first plot
    uniq_dict = {}
    for i in range(len(err_type_dict)):
        uniq_dict[i] = 0

    for key in err_dict:
        err_dict[key] = list(set(err_dict[key]))

        if "No Error" not in err_dict[key]:
            uniq_dict[len(err_dict[key])] += 1
        else:
            if len(err_dict[key]) == 1:
                uniq_dict[0] += 1
            else:
                uniq_dict[len(err_dict[key]) - 1] += 1
    print(uniq_dict)
    print(err_type_dict)
    plt.bar(uniq_dict.keys(), uniq_dict.values(), align="center", alpha=0.7)  # type: ignore
    plt.xlabel("number of unique error(s)")
    plt.ylabel("number of solution(s)")
    plt.title("Histogram of the number of unique error(s)")
    plt.savefig("unique_errors.png")

    plt.bar(err_type_dict.keys(), err_type_dict.values(), align="center", alpha=0.7)  # type: ignore
    plt.xlabel("Types of Error")
    plt.ylabel("number of errors")
    plt.title("Bar plot of the number of errors in each type")
    plt.savefig("type_cnt_errors.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
