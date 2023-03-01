import json, argparse



def create_corpus(args):
    corpus = []
    with open(args.path_to_conceptnet_csv_en, "r", encoding='utf8') as f:
        for line in f.readlines():
            ls = line.strip().split('\t')
            rel = ls[0]
            head = ls[1]
            tail = ls[2]




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="use conceptnet 5.7 csv")
    parser.add_argument(
        "--path_to_conceptnet_csv_en",
        type=str,
        default='../../../conceptnet/assertions-570-en.csv'
    )
    args = parser.parse_args()
    create_corpus(args)
