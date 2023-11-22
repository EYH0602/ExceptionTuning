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
import re
from label import LABELS, to_int_label


def parse_exception(traceback_str: str) -> tuple[str, str]:
    # if empty, assume no error
    if traceback_str == "":
        return "NoError", ""

    # regex to find the error type
    error_type_pattern = r"^\w+Error"
    error_type_match = re.search(error_type_pattern, traceback_str, re.MULTILINE)

    # regex to find the line of code that caused the error
    error_line_pattern = r'File ".*", line \d+, in .*\n\s+(.*)'
    error_line_match = re.search(error_line_pattern, traceback_str, re.MULTILINE)

    error_type = error_type_match.group() if error_type_match else "UnknownError"
    error_line = error_line_match.group(1).strip() if error_line_match else ""

    return error_type, error_line


def get_err(file_path: str) -> tuple[str, str]:
    """read in the give file and parse the exception

    Args:
        file_path (str): path to a stderr dump file

    Returns:
        tuple[str, str]: Error Type, Error Line
    """
    with open(file_path) as f:
        traceback = f.read()

    return parse_exception(traceback)


def process_one_solution(
    fuzz_dir_path, code_dir_path, p: str, s: str, cont: int
) -> tuple[list[int], dict]:
    fuzz_path = pjoin(fuzz_dir_path, p, s, "default")
    if not os.path.exists(fuzz_path):
        return [], {}
    err_path = pjoin(fuzz_path, "stderr")
    errs = [get_err(pjoin(err_path, f)) for f in sorted(os.listdir(err_path))]
    js = {}

    # use first instance as data for faster training
    js["index"] = str(cont)
    js["label"] = to_int_label(errs[0][0]).unwrap()
    js["error"] = errs[0][1]
    with open(pjoin(code_dir_path, p, s), encoding="latin-1") as f:
        js["code"] = f.read()
    return list(map(lambda x: to_int_label(x[0]).unwrap(), errs)), js


def main(
    fuzz_dir_path="fuzz",
    code_dir_path="Project_CodeNet_Python800",
    jobs=80,
):
    # initialize for two plots
    err_dict: dict[str, list[int]] = {}
    err_type_dict: dict[int, int] = {}

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
        results = list(
            tqdm(pool.starmap(process_one_solution, solutions), total=len(solutions))
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

    uniq_keys = []
    uniq_values = []
    for k in uniq_dict:
        if uniq_dict[k] != 0:
            uniq_keys.append(k)
            uniq_values.append(uniq_dict[k])

    plt.bar(uniq_keys, uniq_values, align="center", alpha=0.7, color="blue")  # type: ignore
    plt.xlabel("Number of Unique Error(s)")
    plt.ylabel("Number of Program(s)")
    plt.yscale("log")
    plt.xticks(uniq_keys)
    # plt.title("Histogram of the number of unique error(s)")
    plt.savefig("unique_errors.png", bbox_inches="tight", dpi=300)

    plt.clf()  # clear plt

    plt.figure(figsize=(8, 9.5))
    labels = list(err_type_dict.keys())
    cnts = list(err_type_dict.values())
    plt.bar(labels, cnts, align="center", alpha=0.7, color="blue")
    plt.xlabel("Types of Error")
    plt.ylabel("Number of Error(s)")
    # plt.title("Bar plot of the number of errors in each type")
    plt.xticks(labels)
    plt.yscale("log")
    plt.savefig("type_cnt_errors.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
