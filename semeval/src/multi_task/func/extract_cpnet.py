import configparser, argparse
import json

relation_mapping = dict()


def load_merge_relation(args):
    #config = configparser.ConfigParser()
    #config.read(args)
    with open(args.path_to_merge_relation_file, encoding="utf8") as f:
        for line in f.readlines():
            ls = line.strip().split('/')
            rel = ls[0]
            for l in ls:
                if l.startswith("*"):
                    relation_mapping[l[1:]] = "*" + rel
                else:
                    relation_mapping[l] = rel


def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    :param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s


def extract_english(args):
    """
    Reads original conceptnet csv file and extracts all English relations (head and tail are both English entities) into
    a new file, with the following format for each line: <relation> <head> <tail> <weight>.
    :return:
    """
    #config = configparser.ConfigParser()
    #config.read(path)

    only_english = []
    with open(args.path_to_conceptnet_csv, encoding="utf8") as f:
        for line in f.readlines():
            ls = line.split('\t')
            if ls[2].startswith('/c/en/') and ls[3].startswith('/c/en/'):
                """
                Some preprocessing:
                    - Remove part-of-speech encoding.
                    - Split("/")[-1] to trim the "/c/en/" and just get the entity name, convert all to 
                    - Lowercase for uniformity.
                """
                rel = ls[1].split("/")[-1].lower()
                head = del_pos(ls[2]).split("/")[-1].lower()
                tail = del_pos(ls[3]).split("/")[-1].lower()

                if not head.replace("_", "").replace("-", "").isalpha():
                    continue

                if not tail.replace("_", "").replace("-", "").isalpha():
                    continue

                if rel not in relation_mapping:
                    continue
                rel = relation_mapping[rel]
                if rel.startswith("*"):
                    rel = rel[1:]
                    tmp = head
                    head = tail
                    tail = tmp

                data = json.loads(ls[4])

                only_english.append("\t".join([rel, head, tail, str(data["weight"])]))

    with open(args.path_to_conceptnet_csv_en, "w", encoding="utf8") as f:
        f.write("\n".join(only_english))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="use conceptnet 5.7 csv")
    parser.add_argument(
        "--path_to_conceptnet_csv",
        type=str,
        default='../../../conceptnet/assertions-570.csv'
    )
    parser.add_argument(
        "--path_to_merge_relation_file",
        type=str,
        default='../../../conceptnet/merge_relation.txt'
    )
    parser.add_argument(
        "--path_to_conceptnet_csv_en",
        type=str,
        default='../../../conceptnet/assertions-570-en.csv'
    )
    parser.add_argument(
        "--path_to_paths_cfg",
        type=str,
        default='../../../conceptnet/paths.cfg'
    )
    args = parser.parse_args()

    load_merge_relation(args)
    print(relation_mapping)
    extract_english(args)