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
        return new_id

    def insert_annotation(self, annotation: dict, text_id: str | None = None, url_id: str | None = None) -> str:
        annotation_id = self.client.query("SELECT generateUUIDv4() AS new_id").result_rows[0][0]
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
        logger.debug(f"results_yandex: {results_yandex}")

        results_google = self.client.query(
            "SELECT url, abstract FROM patent_google WHERE url IN %(urls)s",
            parameters={'urls': urls_google}
        ).result_rows

        logger.debug(f"results_google: {results_google}")
        return results_yandex, results_google

    def delete_annotations_by_urls(self, urls: list[str]) -> None:
        annotations_ids = [
            result[0]
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
        annotations_ids_to_url = {ann[0]:ann[1] for ann in annotations}
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
            annotation_id_to_result.setdefault(comp[0], [])
            annotation_id_to_result[comp[0]].append({
                'from_name': "label",
                'to_name': "text",
                'type': 'labels',
                'id': comp[5],
                'value': {
                    'start': comp[1],
                    'end': comp[2],
                    'text': comp[4],
                    'labels': comp[3]
                }
            })
        for rel in relations:
            annotation_id_to_result.setdefault(rel[0], [])
            annotation_id_to_result[rel[0]].append({
                'from_id': rel[1],
                'to_id': rel[2],
                'type': 'relation',
                'direction': rel[4],
                'labels': rel[3]
            })

        for ann_id in annotations_ids:
            annotation_id_to_result[ann_id].append({
                "from_name": "database_id",
                "origin": "manual",
                "to_name": "text",
                "type": "textarea",
                "value": {
                    "text": [ann_id]
                }
            })

        for ann_id, result_data in annotation_id_to_result.items():
            cur_url = annotations_ids_to_url[ann_id]
            result.setdefault(cur_url, [])
            result[cur_url].append(result_data)
        return result

