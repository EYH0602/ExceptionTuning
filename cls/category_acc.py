import matplotlib.pyplot as plt

cat_accs = {
    "0": 0.999,
    "1": 0.9996,
    "10": 0.9192,
    "11": 0.7097,
    "12": 0.0,
    "13": 0.9524,
    "14": 0.0,
    "15": 0.0,
    "17": 0.0,
    "18": 0.0,
    "2": 0.9488,
    "3": 0.4889,
    "4": 0.0,
    "5": 0.7455,
    "6": 0.0,
    "7": 0.9718,
    "8": 0.9894,
    "9": 0.9976,
}


def main():
    accs = []
    for i in range(22):
        if str(i) in cat_accs:
            accs.append(cat_accs[str(i)] * 100)
        else:
            accs.append(0)

    idxs = list(range(22))

    plt.bar(idxs, accs, align="center", alpha=0.7, color="blue")  # type: ignore
    plt.xlabel("Type of Error")
    plt.ylabel("Test Prediction Accuracy")
    plt.xticks(idxs)
    # plt.title("Histogram of the number of unique error(s)")
    plt.savefig("per_category_acc.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
