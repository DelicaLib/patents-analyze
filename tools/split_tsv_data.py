import os
import sys
from collections import Counter
import io
from glob import glob

import click
from numpy.random import permutation
from numpy import split

import pprint
import logging

logging.basicConfig(format="%(message)s")


def calc_distribution(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    distribution = {key: (value / total) * 100 for key, value in counter.items()}
    return distribution


def get_proportion_counter(entity_data: list[str]):
    entity_counter = Counter()
    for doc in entity_data:
        if len(doc) == 0:
            continue
        for word, entity in map(lambda x: str(x).split(" "), doc.split("\n")):
            if entity.startswith("B-"):
                entity_counter[entity[2:]] += 1
    return entity_counter


def print_proportion(entity_counter: Counter):
    print(f"Tokens count: {sum(entity_counter.values())}")
    print("Tokens proportion: ", end="")
    pprint.pprint(calc_distribution(entity_counter))


def parse_tsv(tsv_content: str):
    cur_tsvs = tsv_content.replace("-DOCSTART- -X- O\n", "").replace("-X- _ ", "").replace("O1 ", "").split("\n\n")
    for idx, tsv in enumerate(cur_tsvs, 0):
        if tsv.find("NN") != -1:
            tsv_tokens = tsv.split("\n")
            for i, tokens in enumerate(tsv_tokens, 0):
                tsv_tokens[i] = "\t".join(tokens.split("\t")[::2])
            cur_tsvs[idx] = "\n".join(tsv_tokens)

    tsvs_tmp = set()
    tsvs_result = []
    for tsv in cur_tsvs:
        tmp = tsv.strip()
        if len(tmp) < 2 or tmp not in tsvs_tmp:
            tsvs_tmp.add(tmp)
            tsvs_result.append(tsv)

    return tsvs_result


def load_and_split_tsvs(paths, train_size):
    tsvs = []
    tsvs_split = [[], []]
    for path in paths:
        with io.open(path, mode="r", encoding="utf-8") as f:
            content = f.read()
            cur_tsvs = parse_tsv(content)
            tsvs.extend(cur_tsvs)
            cur_split_tsvs = split(list(permutation(cur_tsvs)), [int(train_size * len(cur_tsvs))])
            tsvs_split[0].extend(cur_split_tsvs[0])
            tsvs_split[1].extend(cur_split_tsvs[1])

    all_counter = get_proportion_counter(tsvs)

    print(f"Count tsv docs: {len(tsvs)}")
    print_proportion(all_counter)
    print(f"Count tsv train docs: {len(tsvs_split[0])}")
    print_proportion(get_proportion_counter(tsvs_split[0]))
    print(f"Count tsv test docs: {len(tsvs_split[1])}")
    print_proportion(get_proportion_counter(tsvs_split[1]))
    return tsvs, tsvs_split


@click.command()
@click.argument('input_path', nargs=1)
@click.argument('output_path', nargs=1)
@click.option(
    '-r', '--recursive',
    is_flag=True, default=False,
    help="Рекурсивно читать все файлы типа -t в input_path"
)
@click.option('-t', '--file_type', type=str, default="tsv", help="Тип файлов")
@click.option(
    '-ts', '--train_size',
    type=float, default=0.9,
    help="Доля обучающей выборки в общей выборке. Число от 0.01 до 0.99"
)
@click.option(
    '--debug',
    is_flag=True, default=False,
    help="Выводить информацию для разработчика. Выводить подробную ошибку при неудачном парсинге файла"
)
def split_tsv(
        input_path: str, output_path: str,
        recursive: bool, file_type: str,
        train_size: float, debug: bool
):
    """Читает файл input_path, делит на обучающую, тестовую выборки
и сохраняет в директорию output_path вместе с общей выборкой.

Если input_path - директория и нет параметра -r, то прочитаются все файлы типа -t внутри этой директории.
Если input_path - директория и есть параметр -r, то прочитают все файлы типа -t внутри этой директории
и внутри всех поддиректорий.
"""
    input_path = input_path.rstrip("/\\")
    if not os.path.isdir(output_path):
        logging.error("output_path должна быть существующей директорией")
        return
    if not os.path.exists(input_path):
        logging.error("input_path не существует")
        return
    tsv_paths = [input_path]
    if not recursive and os.path.isdir(input_path):
        tsv_paths = glob(f"{input_path}/*.{file_type}")
    elif recursive and os.path.isdir(input_path):
        tsv_paths = glob(f"{input_path}/**/*.{file_type}", recursive=True)

    if len(tsv_paths) == 0:
        logging.error(f"в input_path не найдено файлов типа {file_type}")
        return

    if 0.01 > train_size or train_size > 0.99:
        logging.error("train_size имеет некорректное значение. Корректное значение от 0.01 до 0.99")
        return
    try:
        tsv_all, tsv_split = load_and_split_tsvs(tsv_paths, train_size)
    except ValueError as ex:
        logging.error("Не удалось разделить выборку. Возможно, формат данных в файле неправильный")
        if debug:
            raise ex
        return
    with io.open(f"{output_path}/train.tsv", mode="w", encoding="utf-8") as f:
        f.write("\n\n".join(tsv_split[0]).strip())
    with io.open(f"{output_path}/test.tsv", mode="w", encoding="utf-8") as f:
        f.write("\n\n".join(tsv_split[1]).strip())
    with io.open(f"{output_path}/all.tsv", mode="w", encoding="utf-8") as f:
        f.write("\n\n".join(tsv_all).strip())


if __name__ == '__main__':
    split_tsv()
