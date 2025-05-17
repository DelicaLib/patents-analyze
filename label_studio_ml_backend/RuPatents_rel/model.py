import os
from itertools import permutations
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue
import spacy
import clickhouse_connect

MODEL_NER_NAME = os.getenv("MODEL_NER", "ru_patents_ner_tiny")
MODEL_REL_NAME = os.getenv("MODEL_REL", "ru_patents_rel_tiny")

nlp_ner = spacy.load(MODEL_NER_NAME)
nlp_rel = spacy.load(MODEL_REL_NAME)
nlp_rel.add_pipe('sentencizer')

CLICKHOUSE_DB = os.getenv("CLICKHOUSE_DB", "dev")
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "dev")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "dev")
CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))

client = clickhouse_connect.get_client(
    host=CLICKHOUSE_HOST,
    port=CLICKHOUSE_PORT,
    username=CLICKHOUSE_USER,
    password=CLICKHOUSE_PASSWORD,
    database=CLICKHOUSE_DB
)

THRESHOLD = 0.2

class RuPatentsRel(LabelStudioMLBase):
    """Custom ML Backend model
    """


    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")

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
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """
        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

