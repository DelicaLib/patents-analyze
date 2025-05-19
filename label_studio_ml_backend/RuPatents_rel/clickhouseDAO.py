import logging
import os
import clickhouse_connect

logger = logging.getLogger(__name__)


CLICKHOUSE_DB = os.getenv("CLICKHOUSE_DB", "dev")
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "dev")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "dev")
CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))

class ClickhouseDAO:
    def __init__(self, ):
        self.client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            username=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASSWORD,
            database=CLICKHOUSE_DB
        )

    def get_last_patent_google_date(self):
        query_result = self.client.query(f"""
            SELECT publicationDate
            FROM patent_google
            ORDER BY publicationDate DESC
            LIMIT 1
        """).result_rows
        return query_result[0][0] if len(query_result) > 0 else None

    def get_last_patent_yandex_date(self):
        query_result = self.client.query(f"""
            SELECT publishedDate
            FROM patent_yandex
            ORDER BY publishedDate DESC
            LIMIT 1
        """).result_rows
        return query_result[0][0] if len(query_result) > 0 else None

    def insert_raw_text(self, text) -> str:
        new_id = self.client.query("SELECT generateUUIDv4() AS new_id").result_rows[0][0]
        self.client.insert("texts", [(new_id, text)], column_names=['id', 'content'])
        return str(new_id)

    def insert_raw_texts(self, texts: list[str]) -> list[str]:
        if len(texts) == 0:
            return []
        query_result = self.client.query(f"SELECT generateUUIDv4() AS new_id FROM numbers({len(texts)});").result_rows
        new_ids = [str(row[0]) for row in query_result]
        self.client.insert("texts", [(new_id, text) for new_id, text in zip(new_ids, texts)], column_names=['id', 'content'])
        return new_ids

    def insert_annotation(self, annotation: dict, text_id: str | None = None, url_id: str | None = None) -> str:
        annotation_id = str(self.client.query("SELECT generateUUIDv4() AS new_id").result_rows[0][0])
        if text_id is None and url_id is None:
            raise ValueError("Either text_id or url_id must be provided")

        ref_id = text_id if text_id is not None else url_id
        ref_id_name = 'text_id' if text_id is not None else 'url'
        self.client.insert("annotations", [(annotation_id, ref_id)], column_names=['id', ref_id_name])
        labels_list = []
        relations_list = []
        for result in annotation['result']:
            if result['type'] == 'labels':
                labels_list.append([
                    annotation_id, result['value']['start'],
                    result['value']['end'], result['value']['labels'],
                    result['value']['text'], result['id']
                ])
            elif result['type'] == 'relation':
                relations_list.append([
                    annotation_id, result['from_id'],
                    result['to_id'], result['labels'],
                    result['direction']
                ])
        self.client.insert("components", labels_list, column_names=[
            'annotation_id', 'start', 'end', 'labels', 'text', 'token_id'
        ])
        self.client.insert("relations", relations_list, column_names=[
            'annotation_id', 'from_id', 'to_id', 'labels', 'direction'
        ])
        return annotation_id

    def select_patent_by_urls(self, urls_yandex: list[str], urls_google: list[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        results_yandex = self.client.query(
            "SELECT url, abstract FROM patent_yandex WHERE url IN %(urls)s",
            parameters={'urls': urls_yandex}
        ).result_rows
        results_yandex = [result_yandex for result_yandex in results_yandex if result_yandex[1] is not None]
        logger.debug(f"results_yandex: {results_yandex}")

        results_google = self.client.query(
            "SELECT url, abstract FROM patent_google WHERE url IN %(urls)s",
            parameters={'urls': urls_google}
        ).result_rows
        results_google = [result_google for result_google in results_google if result_google[1] is not None]

        logger.debug(f"results_google: {results_google}")
        return results_yandex, results_google

    def select_patent_by_text_ids(self, text_ids: list[str]) -> list[tuple[str, str]]:
        result_texts = self.client.query(
            "SELECT id, content FROM texts WHERE id IN %(ids)s",
            parameters={'ids': text_ids}
        ).result_rows
        result_texts = [(str(result[0]), result[1]) for result in result_texts]
        return result_texts

    def delete_annotations_by_urls(self, urls: list[str]) -> None:
        annotations_ids = [
            str(result[0])
            for result in self.client.query(
                "SELECT id FROM annotations WHERE url IN %(urls)s",
                parameters={'urls': urls}
            ).result_rows
        ]
        _ = self.client.query(
            "DELETE FROM components WHERE annotation_id IN %(ids)s",
            parameters={'ids': annotations_ids}
        ).result_rows
        _ = self.client.query(
            "DELETE FROM relations WHERE annotation_id IN %(ids)s",
            parameters={'ids': annotations_ids}
        ).result_rows
        _ = self.client.query(
            "DELETE FROM annotations WHERE id IN %(ids)s",
            parameters={'ids': annotations_ids}
        ).result_rows

    def select_annotations_by_urls(self, urls: list[str]) -> dict[str, list[dict]]:
        annotations = self.client.query(
            "SELECT id, url FROM annotations WHERE url IN %(urls)s",
            parameters={'urls': urls}
        ).result_rows
        annotations_ids_to_url = {str(ann[0]):ann[1] for ann in annotations}
        annotations_ids = list(annotations_ids_to_url.keys())

        components = self.client.query(
            "SELECT annotation_id, start, end, labels, text, token_id FROM components WHERE annotation_id IN %(ids)s",
            parameters={'ids': annotations_ids}
        ).result_rows
        relations = self.client.query(
            "SELECT annotation_id, from_id, to_id, labels, direction FROM relations WHERE annotation_id IN %(ids)s",
            parameters={'ids': annotations_ids}
        ).result_rows

        result = {}
        annotation_id_to_result = {}
        for comp in components:
            annotation_id_to_result.setdefault(str(comp[0]), [])
            annotation_id_to_result[str(comp[0])].append({
                'from_name': "label",
                'to_name': "text",
                'type': 'labels',
                'id': str(comp[5]),
                'value': {
                    'start': comp[1],
                    'end': comp[2],
                    'text': comp[4],
                    'labels': comp[3]
                }
            })
        for rel in relations:
            annotation_id_to_result.setdefault(str(rel[0]), [])
            annotation_id_to_result[str(rel[0])].append({
                'from_id': str(rel[1]),
                'to_id': str(rel[2]),
                'type': 'relation',
                'direction': rel[4],
                'labels': rel[3]
            })

        for ann_id in annotations_ids:
            annotation_id_to_result.setdefault(ann_id, [])
            annotation_id_to_result[ann_id].append({
                "from_name": "database_id",
                "origin": "manual",
                "to_name": "text",
                "type": "textarea",
                "value": {
                    "text": [str(ann_id)]
                }
            })

        for ann_id, result_data in annotation_id_to_result.items():
            cur_url = annotations_ids_to_url[str(ann_id)]
            result.setdefault(cur_url, [])
            result[cur_url].append(result_data)
        return result

    def select_annotations_by_urls_and_ids(self, urls: list[str], annotation_ids: list[str]) -> list[str]:
        if len(urls) == 0:
            return []
        pairs = [
            (annotation_id, url)
            for url, annotation_id in zip(urls, annotation_ids)
            if isinstance(annotation_id, str)
        ]
        annotations = self.client.query(
            "SELECT id FROM annotations WHERE (id, url) IN %(pairs)s",
            parameters={'pairs': pairs}
        ).result_rows
        annotations_ids = [str(ann[0]) for ann in annotations]
        return annotations_ids

    def select_annotations_by_text_ids_and_ids(self, text_ids: list[str], annotation_ids: list[str]) -> list[str]:
        if len(text_ids) == 0:
            return []
        pairs = [
            (annotation_id, text_id)
            for text_id, annotation_id in zip(text_ids, annotation_ids)
            if isinstance(annotation_id, str)
        ]
        annotations = self.client.query(
            "SELECT id FROM annotations WHERE (id, text_id) IN %(pairs)s",
            parameters={'pairs': pairs}
        ).result_rows
        annotations_ids = [str(ann[0]) for ann in annotations]
        return annotations_ids

    def replace_annotation(self, annotation: dict, annotation_id: str) -> str:
        _ = self.client.query(
            "DELETE FROM components WHERE annotation_id IN %(ids)s",
            parameters={'ids': [annotation_id]}
        ).result_rows
        _ = self.client.query(
            "DELETE FROM relations WHERE annotation_id IN %(ids)s",
            parameters={'ids': [annotation_id]}
        ).result_rows
        labels_list = []
        relations_list = []
        for result in annotation['result']:
            if result['type'] == 'labels':
                labels_list.append([
                    annotation_id, result['value']['start'],
                    result['value']['end'], result['value']['labels'],
                    result['value']['text'], result['id']
                ])
            elif result['type'] == 'relation':
                relations_list.append([
                    annotation_id, result['from_id'],
                    result['to_id'], result['labels'],
                    result['direction']
                ])
        self.client.insert("components", labels_list, column_names=[
            'annotation_id', 'start', 'end', 'labels', 'text', 'token_id'
        ])
        self.client.insert("relations", relations_list, column_names=[
            'annotation_id', 'from_id', 'to_id', 'labels', 'direction'
        ])
        return annotation_id

