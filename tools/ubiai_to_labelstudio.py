import io
import logging
import os
import re
from glob import glob
import json

import click
import uuid
import shortuuid

logging.basicConfig(format="%(message)s", level=logging.INFO)


def ubiai_tsv_to_labelstudio(tsv_file, force: bool):
    with open(tsv_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    documents = []
    text, entities = "", []
    start, label = None, None
    offset = 0

    for i, line in enumerate(lines, 1):
        line = line.strip()

        if line == "":  # Пустая строка
            if text.strip():  # Проверяем, есть ли данные
                documents.append({
                    "data": {"text": text.strip()},
                    "annotations": [{"result": entities}]
                })
            text, entities = "", []  # Сбрасываем данные для нового документа
            offset = 0
            continue
        try:
            token, tag = re.split(r'[\t ]+', line)
        except ValueError as ex:
            logging.error(f"""Неверный формат файла {tsv_file}. 
Строка должна быть в виде 'token tag' или 'token\\ttag'. Строка номер {i} выглядит как: 
{line}""")
            if not force:
                raise ex
            continue

        if tag.startswith("B-"):
            if label:
                entities.append({
                    "value": {
                        "start": start,
                        "end": offset - 1,
                        "text": text[start:offset - 1],
                        "labels": [label]
                    },
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels"
                })
            label = tag[2:]
            start = offset

        elif tag == "O" and label:
            entities.append({
                "value": {
                    "start": start,
                    "end": offset - 1,
                    "text": text[start:offset - 1],
                    "labels": [label]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels"
            })
            label, start = None, None
        text += token + " "
        offset += len(token) + 1  # Учитываем пробел

    # Добавляем последний документ
    if text.strip():
        documents.append({
            "data": {"text": text.strip()},
            "annotations": [{"result": entities}]
        })

    cur_docs = {}

    for doc in documents:
        cur_text = doc["data"]["text"]
        if cur_text.count(" ") <= len(cur_text) / 3:
            cur_docs[cur_text] = doc

    documents = list(cur_docs.values())

    return documents


def ubiai_json_to_labelstudio(json_file, force: bool):
    with io.open(json_file, mode="r", encoding="utf-8") as f:
        content = f.read()
        parsed_content = json.loads(content)

    result = []
    doc_set = set()
    for doc in parsed_content:
        try:
            if doc["document"] in doc_set or doc["document"].count(" ") > len(doc["document"]) / 3:
                continue
            doc_set.add(doc["document"])
            cur_result_doc = {"data": {
                    "text": doc["document"]
                },
                "annotations": [{
                    "result": []
                }]
            }
            token_start_to_id = {}

            for token in doc["tokens"]:
                cur_id = shortuuid.encode(uuid.uuid4())

                token_start_to_id[token["token_start"]] = cur_id
                cur_result_doc["annotations"][0]["result"].append(
                    {
                        "value": {
                            "start": token["start"],
                            "end": token["end"],
                            "text": token["text"],
                            "labels": [token["entityLabel"]]
                        },
                        "id": cur_id,
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels"
                    }
                )

            for relation in doc["relations"]:
                id_parent = token_start_to_id[relation["head"]]
                id_child = token_start_to_id[relation["child"]]
                cur_result_doc["annotations"][0]["result"].append(
                    {
                        "from_id": id_parent,
                        "to_id": id_child,
                        "type": "relation",
                        "direction": "right",
                        "labels": [
                            relation["relationLabel"]
                        ]
                    }
                )
        except KeyError as ex:
            logging.error(f"""Неверный формат файла {json_file}. 
            Отсутствует ключ '{ex.args[0]}'""")
            if not force:
                raise ex
            continue
        result.append(cur_result_doc)
    return result


def get_input_files(ctx: click.Context, file_type: str):
    input_path = ctx.obj["input_path"]
    recursive = ctx.obj["recursive"]
    input_paths = [input_path]
    if not recursive and os.path.isdir(input_path):
        input_paths = glob(f"{input_path}/*.{file_type}")
    elif recursive and os.path.isdir(input_path):
        input_paths = glob(f"{input_path}/**/*.{file_type}", recursive=True)

    if len(input_paths) == 0:
        logging.error(f"в input_path не найдено файлов типа {file_type}")
        raise ValueError(f"в input_path не найдено файлов типа {file_type}")

    return input_paths


def convert_procces(ctx: click.Context, file_type: str, covert_func):
    debug = ctx.obj["debug"]
    output_dir = ctx.obj["output_path"]
    force = ctx.obj["force"]
    try:
        input_files = get_input_files(ctx, file_type)
    except ValueError as ex:
        logging.error("Проблема с конвертацией.")
        if debug:
            logging.exception(ex)
        return
    results = []
    output_file_paths_dict = {}
    for input_file in input_files:
        new_file_path = os.path.basename(input_file).split(".")[0] + ".json"
        output_file = "/".join([output_dir.strip("/\\"), new_file_path])
        if new_file_path in output_file_paths_dict:
            output_file = "/".join([output_dir.strip("/\\"), str(output_file_paths_dict[new_file_path]) + new_file_path])
        else:
            output_file_paths_dict[new_file_path] = 0
        output_file_paths_dict[new_file_path] += 1
        try:
            results.append([covert_func(input_file, force), output_file])
        except ValueError as ex:
            logging.error("Проблема с конвертацией.")
            if debug:
                logging.exception(ex)
            return
    result_docs_count = 0
    for result, output_path in results:
        result_docs_count += len(result)
        with io.open(output_path, mode="w", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False))
            logging.info(f"Сохранено {output_path}. Количество документов: {len(result)}")
    logging.info(f"Итоговое количество документов: {result_docs_count}")


@click.group()
@click.argument('input_path', nargs=1)
@click.argument('output_path', nargs=1)
@click.option(
    '-r', '--recursive',
    is_flag=True, default=False,
    help="Рекурсивно читать все файлы типа -t в input_path"
)
@click.option(
    '-f', '--force',
    is_flag=True, default=False,
    help="Обработать всё, не обращая внимания на ошибки"
)
@click.option(
    '--debug',
    is_flag=True, default=False,
    help="Выводить информацию для разработчика. Выводить подробную ошибку при неудачном парсинге файла"
)
@click.pass_context
def cli(
    ctx: click.Context, input_path: str,
    output_path: str, recursive: bool,
    debug: bool, force: bool
):
    input_path = input_path.rstrip("/\\")
    if not os.path.isdir(output_path):
        logging.error("output_path должна быть существующей директорией")
        return
    if not os.path.exists(input_path):
        logging.error("input_path не существует")
        return
    ctx.ensure_object(dict)
    ctx.obj["input_path"] = input_path
    ctx.obj["recursive"] = recursive
    ctx.obj["output_path"] = output_path
    ctx.obj["force"] = force
    ctx.obj["debug"] = debug


@cli.command("tsv")
@click.option('-t', '--file_type', type=str, default="tsv", help="Тип файлов", show_default=True)
@click.pass_context
def tsv_to_labelstudio_command(ctx: click.Context, file_type: str):
    convert_procces(ctx, file_type, ubiai_tsv_to_labelstudio)


@cli.command("json")
@click.option('-t', '--file_type', type=str, default="json", help="Тип файлов", show_default=True)
@click.pass_context
def json_to_labelstudio_command(ctx: click.Context, file_type: str):
    convert_procces(ctx, file_type, ubiai_json_to_labelstudio)


if __name__ == '__main__':
    cli()
