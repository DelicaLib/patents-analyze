import logging
import os
from itertools import permutations
from typing import List, Dict, Optional

from label_studio_ml.exceptions import AnswerException
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue
import spacy
from datetime import date
import threading
import requests
from dateutil.relativedelta import relativedelta
from clickhouseDAO import ClickhouseDAO
from urllib.parse import urlparse, urlunparse

MODEL_NER_NAME = os.getenv("MODEL_NER", "ru_patents_ner_tiny")
MODEL_REL_NAME = os.getenv("MODEL_REL", "ru_patents_rel_tiny")

nlp_ner = spacy.load(MODEL_NER_NAME)
nlp_rel = spacy.load(MODEL_REL_NAME)
nlp_rel.add_pipe('sentencizer')

PARSER_HOST = os.getenv("PARSER_HOST", "localhost")
PARSER_PORT = os.getenv("PARSER_PORT", "5000")

THRESHOLD = 0.2


logger = logging.getLogger(__name__)


class RuPatentsRel(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def __init__(self, project_id: Optional[str] = None, label_config=None):
        super().__init__(project_id, label_config)
        self.clickhouse = ClickhouseDAO()

    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        from_name, to_name, value = self.label_interface.get_first_tag_occurence('Labels', 'Text')
        predictions = []
        for task in tasks:
            text = self.preload_task_data(task, task['data'][value])
            docs = nlp_ner.pipe([text], disable=["tagger", "parser"])
            for doc in docs:
                for name, proc in nlp_rel.pipeline:
                    doc = proc(doc)
                predictions.append(PredictionValue(
                    result=self._prepare_annotation_from_doc(doc, from_name, to_name),
                    model_version=self.get('model_version')
                ))
        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        return

    def annotate_texts(self, texts: list[dict]) -> list[dict]:
        results = []
        texts = [i['text'] for i in texts]
        docs = nlp_ner.pipe(texts, disable=["tagger", "parser"])
        for doc in docs:
            for name, proc in nlp_rel.pipeline:
                doc = proc(doc)
            text = doc.text
            text_database_id = self.clickhouse.insert_raw_text(text)
            annotation = {
                "result": self._prepare_annotation_from_doc(doc, "label", "text")
            }
            annotation_database_id = self.clickhouse.insert_annotation(annotation, text_database_id)
            annotation["result"].append(
                {
                    "from_name": "database_id",
                    "origin": "manual",
                    "to_name": "text",
                    "type": "textarea",
                    "value": {
                        "text": [annotation_database_id]
                    }
                }
            )
            results.append({
                "data": {
                    "text": text
                },
                "meta": {
                    "database_id": text_database_id
                },
                "annotations": [annotation]
            })
        return results

    def get_annotation_by_url(self, urls: list[dict]) -> list[dict]:
        results = []
        yandex_patents, google_patents, urls_from_database = self._preprocess_patents_from_url(urls, run_parse=False)
        url_to_annotations = self.clickhouse.select_annotations_by_urls(urls_from_database)
        all_patents = yandex_patents + google_patents
        url_to_result = self._generate_labelstuio_task(all_patents, url_to_annotations)
        for url in urls:
            results.append(url_to_result[url['url']])
        return results

    def annotate_patents_from_url(self, urls: list[dict], annotate_if_exist: bool) -> list[dict]:
        results = []
        yandex_patents, google_patents, urls_from_database = self._preprocess_patents_from_url(urls, run_parse=True)
        all_patents = yandex_patents + google_patents
        if not annotate_if_exist:
            url_to_annotations = self.clickhouse.select_annotations_by_urls(urls_from_database)
            url_to_result = self._generate_labelstuio_task(all_patents, url_to_annotations)
            all_patents_text = [
                patent[1]
                for patent in all_patents
                if len(url_to_result.get(patent[0], {}).get("annotations", [])) == 0
            ]
        else:
            all_patents_text = [patent[1] for patent in all_patents]
            url_to_result = {}
        docs = nlp_ner.pipe(all_patents_text, disable=["tagger", "parser"])
        for i, doc in enumerate(docs):
            cur_url = all_patents[i][0]
            for name, proc in nlp_rel.pipeline:
                doc = proc(doc)
            text = doc.text
            annotation = {
                "result": self._prepare_annotation_from_doc(doc, "label", "text")
            }
            annotation_database_id = self.clickhouse.insert_annotation(annotation, url_id=cur_url)
            annotation["result"].append(
                {
                    "from_name": "database_id",
                    "origin": "manual",
                    "to_name": "text",
                    "type": "textarea",
                    "value": {
                        "text": [annotation_database_id]
                    }
                }
            )
            url_to_result[cur_url] = {
                "data": {
                    "text": text
                },
                "meta": {
                    "url": cur_url
                },
                "annotations": [annotation]
            }
        logger.debug(url_to_result)
        for url in urls:
            results.append(url_to_result[url['url']])

        return results

    def insert_annotations(self, data: list[dict]):
        pass

    def _generate_labelstuio_task(self, patents_from_db, url_to_annotations) -> dict:
        url_to_task = {}
        for patent in patents_from_db:
            text = patent[1]
            url = patent[0]
            annotations = []
            for annotation in url_to_annotations.get(url, []):
                annotations.append({"result": annotation})
            url_to_task[url] = {
                "data": {
                    "text": text
                },
                "meta": {
                    "url": url
                },
                "annotations": annotations
            }
        return url_to_task

    def _preprocess_patents_from_url(self, urls: list[dict], *, run_parse: bool = False)\
            -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
        urls_list = [i['url'] for i in urls]
        valid_urls_yandex, valid_urls_google, invalid_urls = self._validate_urls(urls_list)
        if len(invalid_urls) > 0:
            raise AnswerException(
                400, 'Один или несколько url невалидны. Примеры валидных url: '
                     'https://patents.google.com/patent/RU2752521C2 '
                     'https://patents.google.com/patent/RU2752521C2/ru '
                     'https://yandex.ru/patents/search/doc/RU221619U1_20231115.'
                     'Подробности в result',
                {
                    "valid_urls": valid_urls_yandex + valid_urls_google,
                    "invalid_urls": invalid_urls,
                    "traceback": ""
                }
            )
        yandex, google = self.clickhouse.select_patent_by_urls(valid_urls_yandex, valid_urls_google)
        urls_from_database = set([data[0] for data in yandex] + [data[0] for data in google])
        not_found_urls = set(urls_list) - urls_from_database
        if len(not_found_urls) > 0:
            msg = 'Один или несколько url не найдены в базе данных.'
            if run_parse:
                threading.Thread(target=self._parse_google_patents).start()
                threading.Thread(target=self._parse_yandex_patents).start()
                msg = 'Один или несколько url не найдены в базе данных. Запущен процесс парсинга. ' \
                    'Возможно, они скоро появятся'
            raise AnswerException(
                404, msg,
                {
                    "valid_urls": list(urls_from_database),
                    "invalid_urls": list(not_found_urls),
                    "traceback": ""
                }
            )
        return yandex, google, list(urls_from_database)

    def _prepare_annotation_from_doc(self, doc, from_name, to_name) -> list:
        result = []
        for ent in doc.ents:
            result.append({
                'from_name': from_name,
                'to_name': to_name,
                'type': 'labels',
                'id': ent.start,
                'value': {
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'text': ent.text,
                    'labels': [ent.label_]
                }
            })
        for sent in doc.sents:
            for entity, entity_b in permutations(sent.ents, 2):
                rel_dict = doc._.rel.get((entity.start, entity_b.start))
                if rel_dict is not None:
                    cur_max_relation = max(rel_dict.items(), key=lambda x: x[1])
                    if cur_max_relation[1] >= THRESHOLD:
                        result.append({
                            'from_id': entity.start,
                            'to_id': entity_b.start,
                            'type': 'relation',
                            'direction': 'right',
                            'labels': [cur_max_relation[0]]
                        })
        return result

    def _parse_google_patents(self):
        parser_url = f"http://{PARSER_HOST}:{PARSER_PORT}/graphql"
        query = """
        mutation EnqueueGooglePatents($input: GoogleSearchInput!) {
            enqueueGooglePatents(input: $input)
        }"""
        today = date.today()
        tmp_clickhouse = ClickhouseDAO()
        date_from = tmp_clickhouse.get_last_patent_google_date()
        variables = {
            "input": {
            "countries": ["RU"],
            "dateTo": today.isoformat(),
            "dateFrom": date_from if date_from is not None else (today - relativedelta(months=1)).isoformat(),
          }
        }
        requests.post(parser_url, json={"query": query, "variables": variables})

    def _parse_yandex_patents(self):
        parser_url = f"http://{PARSER_HOST}:{PARSER_PORT}/graphql"
        query = """
        mutation EnqueueYandexPatents($input: YandexSearchInput!) {
            enqueueYandexPatents(input: $input)
        }"""
        today = date.today()
        tmp_clickhouse = ClickhouseDAO()
        date_from = tmp_clickhouse.get_last_patent_google_date()
        variables = {
            "input": {
            "countries": [
                "RussianFederation",
                "SovietUnion"
            ],
            "dateTo": today.isoformat(),
            "dateFrom": date_from if date_from is not None else (today - relativedelta(months=1)).isoformat(),
            "parserSettings": {
                "perPage": 50
            }
          }
        }
        requests.post(parser_url, json={"query": query, "variables": variables})

    def _validate_urls(self, urls: list[str]) -> tuple[list, list, list]:
        valid_urls_yandex = []
        valid_urls_google = []
        invalid_urls = []
        for url in urls:
            parsed = urlparse(url)
            if parsed.scheme == "https" and parsed.netloc in {"patents.google.com", "yandex.ru"}:
                if parsed.netloc == "patents.google.com":
                    if not parsed.path.startswith("/patent/RU"):
                        invalid_urls.append(url)
                    elif parsed.path.endswith("/ru"):
                        valid_urls_google.append(url)
                    else:
                        invalid_urls.append(url)
                else:
                    if not parsed.path.startswith("/patents/doc/RU"):
                        invalid_urls.append(url)
                    else:
                        valid_urls_yandex.append(url)
            else:
                invalid_urls.append(url)

        return valid_urls_yandex, valid_urls_google, invalid_urls
