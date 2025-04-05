import logging
import os
from collections import Counter
import io
from glob import glob
import json

import click
from numpy.random import permutation
from numpy import split

from pydantic import BaseModel, ValidationError, model_validator
from enum import Enum
import pprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


class Entity(Enum):
    COMPONENT = "COMPONENT"
    SYSTEM = "SYSTEM"
    ATTRIBUTE = "ATTRIBUTE"


class RelationLabel(Enum):
    PART_OF = "PART-OF"
    LOCATED_AT = "LOCATED-AT"
    CONNECTED_WITH = "CONNECTED-WITH"
    ATTRIBUTE_FOR = "ATTRIBUTE-FOR"
    IN_MANNER_OF = "IN-MANNER-OF"


class Token(BaseModel):
    text: str
    start: int
    end: int
    token_start: int
    token_end: int
    entityLabel: Entity
    propertiesList: list
    commentsList: list

    @model_validator(mode='before')
    def check_passwords_match(self):
        if "propertiesList" not in self:
            self["propertiesList"] = []
        if "commentsList" not in self:
            self["commentsList"] = []
        return self


class Relation(BaseModel):
    child: int
    head: int
    relationLabel: RelationLabel
    propertiesList: list
    commentsList: list

    @model_validator(mode='before')
    def check_passwords_match(self):
        if "propertiesList" not in self:
            self["propertiesList"] = []
        if "commentsList" not in self:
            self["commentsList"] = []
        if self["relationLabel"] == "CINNECTED-WITH":
            self["relationLabel"] = "CONNECTED-WITH"
        if self["relationLabel"] == "CONNECTED-WITHCD":
            self["relationLabel"] = "CONNECTED-WITH"
        if self["relationLabel"] == "LOCATED-OF":
            self["relationLabel"] = "LOCATED-AT"
        if self["relationLabel"] == "HAS-ATTRIBUTE":
            self["relationLabel"] = "ATTRIBUTE-FOR"
            self["head"], self["child"] = self["child"], self["head"]
        return self


class Patent(BaseModel):
    documentName: str
    document: str
    tokens: list[Token]
    relations: list[Relation]

    @model_validator(mode='before')
    def remove_overlapping_entities(self):
        fixed_tokens = []
        existing_spans = set()
        removed_token_ids = set()

        for token in self["tokens"]:
            token_range = range(token["start"], token["end"])
            if any(i in existing_spans for i in token_range):
                removed_token_ids.add(token["token_start"])
                continue

            fixed_tokens.append(token)
            existing_spans.update(list(token_range))

        self["tokens"] = fixed_tokens

        self["relations"] = [
            relation for relation in self["relations"]
            if relation["child"] not in removed_token_ids and relation["head"] not in removed_token_ids
        ]

        return self


def calc_distribution(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    distribution = {key: (value / total) * 100 for key, value in counter.items()}
    return distribution


def check_proportion(relations_data: list):
    relation_counter = Counter()
    for value in RelationLabel:
        relation_counter[value.value] = 0
    component_counter = Counter()
    for value in Entity:
        component_counter[value.value] = 0

    for cur_data in relations_data:
        for cur_relation in cur_data["relations"]:
            relation_counter[cur_relation["relationLabel"]] += 1
        for cur_relation in cur_data["tokens"]:
            component_counter[cur_relation["entityLabel"]] += 1

    print(f"Relation count: {sum(relation_counter.values())}")
    print("Relation proportion: ", end="")
    pprint.pprint(calc_distribution(relation_counter))

    print(f"Tokens proportion: {sum(component_counter.values())}")
    print("Tokens proportion: ", end="")
    pprint.pprint(calc_distribution(component_counter))


def check_struct(jsons: tuple):
    results = []
    broken_count = 0
    for parsed_content in jsons:
        parsed_patents = []
        for patent in parsed_content:
            if "tokens" not in patent or "relations" not in patent:
                broken_count += 1
            else:
                try:
                    result = Patent(**patent)
                    parsed_patents.append(result.model_dump(mode="json"))
                except ValidationError as ex:
                    broken_count += 1
        pprint.pprint(len(parsed_patents))
        check_proportion(parsed_patents)
        results.append(parsed_patents)
    print(f"Broken count: {broken_count}")
    return results


def smart_split_json(data: list, train_size):
    relations = [set(rel["relationLabel"] for rel in doc["relations"]) if "relations" in doc else {} for doc in data]
    mlb = MultiLabelBinarizer()
    relation_matrix = mlb.fit_transform(relations)
    test_size = (1 - train_size) / 0.5
    temp_docs, test_docs = train_test_split(
        data, test_size=test_size, stratify=relation_matrix, random_state=42
    )
    relations = [set(rel["relationLabel"] for rel in doc["relations"]) if "relations" in doc else {} for doc in
                 temp_docs]
    mlb = MultiLabelBinarizer()
    relation_matrix = mlb.fit_transform(relations)
    test_size = (len(test_docs) / len(temp_docs))
    train_docs, dev_docs = train_test_split(
        temp_docs, test_size=test_size, stratify=relation_matrix, random_state=42
    )

    print(f"Patents rel train count: {len(train_docs)}")
    print(f"Patents rel test count: {len(test_docs)}")
    print(f"Patents rel dev count: {len(dev_docs)}")

    check_proportion(train_docs)
    check_proportion(test_docs)
    check_proportion(dev_docs)
    return train_docs, test_docs, dev_docs


def load_and_split_jsons(paths: list[str], train_size: float):
    patents = []
    failed_files = []
    patents_split = [[], [], []]
    texts_set = set()
    for path in paths:
        with io.open(path, mode="r", encoding="utf-8") as f:
            content = f.read()
            try:
                parsed_content_raw1 = json.loads(content)
                parsed_content_raw = []
                for cur_doc in parsed_content_raw1:
                    if "document" not in cur_doc:
                        continue
                    if cur_doc["document"] in texts_set:
                        continue
                    texts_set.add(cur_doc["document"])
                    patents.append(cur_doc)
                    parsed_content_raw.append(cur_doc)
                if len(parsed_content_raw) == 0:
                    continue
                try:
                    parsed_content = list(permutation(parsed_content_raw))
                    if 25 >= len(parsed_content) >= 2:
                        patents_split[0].extend(parsed_content[2:])
                        patents_split[1].append(parsed_content[0])
                        patents_split[2].append(parsed_content[1])
                    else:
                        parsed_content = smart_split_json(parsed_content_raw, train_size)
                        patents_split[0].extend(parsed_content[0])
                        patents_split[1].extend(parsed_content[1])
                        patents_split[2].extend(parsed_content[2])
                except ValueError as ex:
                    parsed_content = split(parsed_content, [int(train_size * len(parsed_content))])
                    patents_split[0].extend(parsed_content[0])
                    parsed_content = split(parsed_content[1], [int(0.5 * len(parsed_content[1]))])
                    patents_split[1].extend(parsed_content[0])
                    patents_split[2].extend(parsed_content[1])
                    print(f"ValueError with parsing {path}: {ex}")
            except json.decoder.JSONDecodeError as e:
                print(f"Fail to parse {path} because {e}")
                failed_files.append(path)

    print(f"Patents rel count: {len(patents)}")
    print(f"Patents rel train count: {len(patents_split[0])}")
    print(f"Patents rel test count: {len(patents_split[1])}")
    print(f"Patents rel dev count: {len(patents_split[2])}")

    return patents, failed_files, patents_split


@click.command()
@click.argument('input_path', nargs=1)
@click.argument('output_path', nargs=1)
@click.option(
    '-r', '--recursive',
    is_flag=True, default=False,
    help="Рекурсивно читать все файлы типа -t в input_path"
)
@click.option('-t', '--file_type', type=str, default="json", help="Тип файлов")
@click.option(
    '-ts', '--train_size',
    type=float, default=0.8,
    help="Доля обучающей выборки в общей выборке. Число от 0.01 до 0.99"
)
@click.option(
    '--debug',
    is_flag=True, default=False,
    help="Выводить информацию для разработчика. Выводить подробную ошибку при неудачном парсинге файла"
)
def split_jsons(
        input_path: str, output_path: str,
        recursive: bool, file_type: str,
        train_size: float, debug: bool
):
    """Читает файл input_path, делит на обучающую, тестовую и dev выборки
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
    json_paths = [input_path]
    if not recursive and os.path.isdir(input_path):
        json_paths = glob(f"{input_path}/*.{file_type}")
    elif recursive and os.path.isdir(input_path):
        json_paths = glob(f"{input_path}/**/*.{file_type}", recursive=True)

    if len(json_paths) == 0:
        logging.error(f"в input_path не найдено файлов типа {file_type}")
        return

    if 0.01 > train_size or train_size > 0.99:
        logging.error("train_size имеет некорректное значение. Корректное значение от 0.01 до 0.99")
        return
    try:
        patents, failed_files, patents_split = load_and_split_jsons(json_paths, train_size)
        patents, *patents_split = check_struct((patents, *patents_split))
    except ValueError as ex:
        logging.error("Не удалось разделить выборку. Возможно, формат данных в файле неправильный")
        if debug:
            raise ex
        return

    with io.open(f"{output_path}/all.json", mode="w", encoding="utf-8") as f:
        f.write(json.dumps(patents, ensure_ascii=False))
    with io.open(f"{output_path}/train.json", mode="w", encoding="utf-8") as f:
        f.write(json.dumps(list(patents_split[0]), ensure_ascii=False))
    with io.open(f"{output_path}/test.json", mode="w", encoding="utf-8") as f:
        f.write(json.dumps(list(patents_split[1]), ensure_ascii=False))
    with io.open(f"{output_path}/dev.json", mode="w", encoding="utf-8") as f:
        f.write(json.dumps(list(patents_split[2]), ensure_ascii=False))


if __name__ == "__main__":
    split_jsons()
