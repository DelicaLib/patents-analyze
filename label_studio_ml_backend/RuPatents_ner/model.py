from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue
import spacy

nlp = spacy.load("ru_patents_ner")

class RuPatentsNer(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
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
            docs = nlp.pipe([text], disable=["tagger", "parser"])
            entities = []
            for doc in docs:
                for ent in doc.ents:
                    entities.append({
                        'from_name': from_name,
                        'to_name': to_name,
                        'type': 'labels',
                        'value': {
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'text': ent.text,
                            'labels': [ent.label_]
                        }
                    })
                predictions.append(PredictionValue(
                    result=entities,
                    model_version=self.get('model_version')
                ))
        return ModelResponse(predictions=predictions, model_version=self.get("model_version"))

    
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

