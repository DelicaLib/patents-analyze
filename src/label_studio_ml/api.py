import hmac
import logging
import os

from flask import Flask, request, jsonify, Response
from flasgger import Swagger

from RuPatents_rel.model import RuPatentsRel
from .response import ModelResponse
from .model import LabelStudioMLBase
from .exceptions import exception_handler

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Label Studio ML API",
        "description": "API for model serving in Label Studio ML Backend",
        "version": "1.0"
    },
    "basePath": "/",
    "schemes": ["http"],
}

logger = logging.getLogger(__name__)

_server = Flask(__name__)
MODEL_CLASS = LabelStudioMLBase
BASIC_AUTH = None

_server.config['SWAGGER'] = {
    "definitions": {
        "ResultLabel": {
            "type": "object",
            "properties": {
                "from_name": {"type": "string", "enum": ["label"], "example": "label"},
                "id": {"type": "string"},
                "to_name": {"type": "string", "enum": ["text"], "example": "text"},
                "type": {"type": "string", "enum": ["labels"], "example": "labels"},
                "value": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                        "text": {"type": "string"},
                        "labels": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["start", "end", "text", "labels"],
                },
            },
            "required": ["from_name", "id", "to_name", "type", "value"],
        },
        "ResultRelation": {
            "type": "object",
            "properties": {
                "from_id": {"type": "string"},
                "to_id": {"type": "string"},
                "type": {"type": "string", "enum": ["relation"], "example": "relation"},
                "direction": {"type": "string", "enum": ["right", "left", "bi"], "example": "right"},
                "labels": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["from_id", "to_id", "type", "direction", "labels"],
        },
        "ResultDatabaseID": {
            "type": "object",
            "properties": {
                "from_name": {"type": "string", "enum": ["database_id"], "example": "database_id"},
                "to_name": {"type": "string", "enum": ["text"], "example": "text"},
                "type": {"type": "string", "enum": ["textarea"], "example": "textarea"},
                "origin": {"type": "string", "enum": ["manual"], "example": "manual"},
                "value": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["text"],
                },
            },
            "required": ["from_name", "origin", "to_name", "type", "value"],
        },
        'AnnotationResponseItem': {
            'type': 'object',
            'required': ['annotations', 'data', 'meta'],
            'properties': {
                'annotations': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'result': {
                                'type': 'array',
                                'items': {
                                    'oneOf': [
                                        {'$ref': '#/definitions/ResultLabel'},
                                        {'$ref': '#/definitions/ResultRelation'},
                                        {'$ref': '#/definitions/ResultDatabaseID'}
                                    ]
                                }
                            }
                        }
                    }
                },
                'data': {
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string'}
                    }
                },
                'meta': {
                    'type': 'object',
                    'properties': {
                        'database_id': {'type': 'string'},
                        'url': {'type': 'string'}
                    }
                }
            }
        },
        'AnnotationRequestItem': {
            'type': 'object',
            'required': ['annotations', 'data', 'meta'],
            'properties': {
                'annotations': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'result': {
                                'type': 'array',
                                'items': {
                                    'oneOf': [
                                        {'$ref': '#/definitions/ResultLabel'},
                                        {'$ref': '#/definitions/ResultRelation'},
                                        {'$ref': '#/definitions/ResultDatabaseID'}
                                    ]
                                }
                            }
                        }
                    }
                },
                'data': {
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string'}
                    }
                },
                'meta': {
                    'type': 'object',
                    'properties': {
                        'database_id': {'type': 'string'},
                        'url': {'type': 'string'}
                    }
                }
            },
            'additionalProperties': True
        },
        'ErrorResponse': {
            'type': 'object',
            'properties': {
                'status': {'type': 'integer'},
                'detail': {'type': 'string'},
                'request': {'type': 'object'},
                'result': {
                    'type': 'object',
                    'properties': {
                        'traceback': {'type': 'string'},
                        'valid_urls': {
                            'type': 'array',
                            'items': {'type': 'string'}
                        },
                        'invalid_urls': {
                            'type': 'array',
                            'items': {'type': 'string'}
                        }
                    }
                }
            }
        }
    }
}


def init_app(model_class, basic_auth_user=None, basic_auth_pass=None):
    global MODEL_CLASS
    global BASIC_AUTH

    if not issubclass(model_class, LabelStudioMLBase):
        raise ValueError('Inference class should be the subclass of ' + LabelStudioMLBase.__class__.__name__)

    MODEL_CLASS = model_class
    basic_auth_user = basic_auth_user or os.environ.get('BASIC_AUTH_USER')
    basic_auth_pass = basic_auth_pass or os.environ.get('BASIC_AUTH_PASS')
    if basic_auth_user and basic_auth_pass:
        BASIC_AUTH = (basic_auth_user, basic_auth_pass)

    return _server


@_server.route('/predict', methods=['POST'])
@exception_handler
def _predict():
    """
        Обработать задачи с помощью нейросетевой модели
        ---
        tags:
          - LabelStudioML
        parameters:
          - name: body
            in: body
            required: true
            schema:
              type: object
        responses:
          200:
            description: Успешная обработка
            schema:
              type: object
              properties:
                results:
                  type: object
                  properties:
                    model_version:
                      type: string
                    predictions:
                      type: array
                      items:
                        type: object
                        properties:
                          model_version:
                            type: string
                          score:
                            type: double
                          result:
                            type: array
                            items:
                              type: object
                    context:
                      type: object
        """
    data = request.json
    tasks = data.get('tasks')
    label_config = data.get('label_config')
    project = str(data.get('project'))
    project_id = project.split('.', 1)[0] if project else None
    params = data.get('params', {})
    context = params.pop('context', {})

    model = MODEL_CLASS(project_id=project_id,
                        label_config=label_config)

    # model.use_label_config(label_config)

    response = model.predict(tasks, context=context, **params)

    # if there is no model version we will take the default
    if isinstance(response, ModelResponse):
        if not response.has_model_version():
            mv = model.model_version
            if mv:
                response.set_version(str(mv))
        else:
            response.update_predictions_version()

        response = response.model_dump()

    res = response
    if res is None:
        res = []

    if isinstance(res, dict):
        res = response.get("predictions", response)

    return jsonify({'results': res})


@_server.route('/setup', methods=['POST'])
@exception_handler
def _setup():
    """
        Настроить модель
        ---
        tags:
          - LabelStudioML
        parameters:
          - name: body
            in: body
            required: true
            schema:
              type: object
              required:
                - project
                - schema
              properties:
                project:
                  type: string
                  description: id проекта
                schema:
                  type: string
                  description: Label config (XML)
                hostname:
                  type: string
                  description: hostname of the LabelStudio app
                access_token:
                  type: string
                extra_params:
                  type: string
        responses:
          200:
            description: Версия модели
            schema:
              type: object
              properties:
                model_version:
                  type: string
        """
    data = request.json
    project_id = data.get('project').split('.', 1)[0]
    label_config = data.get('schema')
    extra_params = data.get('extra_params')
    model = MODEL_CLASS(project_id=project_id,
                        label_config=label_config)

    if extra_params:
        model.set_extra_params(extra_params)

    model_version = model.get('model_version')
    return jsonify({'model_version': model_version})


TRAIN_EVENTS = (
    'ANNOTATION_CREATED',
    'ANNOTATIONS_CREATED',
    'ANNOTATION_UPDATED',
    'ANNOTATION_DELETED',
    'ANNOTATIONS_DELETED',
    'TASKS_CREATED',
    'START_TRAINING'
)


@_server.route('/health', methods=['GET'])
@_server.route('/', methods=['GET'])
@exception_handler
def health():
    """
       Проверка работоспособности
       ---
       tags:
         - Health
       responses:
         200:
           description: Сервер работает и модель настроена
           schema:
             type: object
             properties:
               status:
                 type: string
                 example: UP
               model_class:
                 type: string
                 example: LabelStudioMLBase
       """
    return jsonify({
        'status': 'UP',
        'model_class': MODEL_CLASS.__name__
    })


@_server.route('/api/annotation/text', methods=['POST'])
@exception_handler
def annotate_text():
    """
        Обработать текст с помощью нейросетевой модели
        ---
        tags:
            - API
        parameters:
            - name: body
              in: body
              required: true
              schema:
                  type: array
                  items:
                      type: object
                      properties:
                          text:
                              type: string
                              description: Текст реферата патента
        responses:
            200:
                description: Размеченные данные в формате labelstudio
                schema:
                    type: array
                    items:
                        $ref: '#/definitions/AnnotationResponseItem'
    """
    data = request.json

    model = RuPatentsRel()
    result = model.annotate_texts(data)

    return jsonify(result)


@_server.route('/api/select/annotation/from_url', methods=['POST'])
@exception_handler
def annotate_patents_from_url():
    """
        Получить разметку в формате labelstudio по ссылкам на патенты
        ---
        tags:
            - API
        parameters:
            - name: body
              in: body
              required: true
              schema:
                  type: array
                  items:
                      type: object
                      properties:
                          url:
                              type: string
                              description: Ссылка на патент
        responses:
            200:
                description: Размеченные данные в формате labelstudio
                schema:
                    type: array
                    items:
                        $ref: '#/definitions/AnnotationResponseItem'
            400:
                description: Неверный формат данных
                schema:
                    $ref: '#/definitions/ErrorResponse'
            404:
                description: Некоротые URL не найдены
                schema:
                    $ref: '#/definitions/ErrorResponse'

    """
    data = request.json

    model = RuPatentsRel()
    result = model.get_annotation_by_url(data)

    return jsonify(result)


@_server.route('/api/annotation/from_url', methods=['POST'])
@exception_handler
def get_patents_from_url():
    """
        Обработать патенты с помощью нейросетевой модели
        ---
        tags:
            - API
        parameters:
            - name: body
              in: body
              required: true
              schema:
                  type: object
                  properties:
                      items:
                          type: array
                          items:
                              type: object
                              properties:
                                  url:
                                      type: string
                                      description: Ссылка на патент
                              required:
                                  - url
                      annotate_if_exist:
                          type: boolean
                          description: Создать новую аннотацию, если аннотации уже присутствуют
                  required:
                      - items
                      - annotate_if_exist
        responses:
            200:
                description: Размеченные данные в формате labelstudio
                schema:
                    type: array
                    items:
                        $ref: '#/definitions/AnnotationResponseItem'
            400:
                description: Неверный формат данных
                schema:
                    $ref: '#/definitions/ErrorResponse'
            404:
                description: Некоротые URL не найдены
                schema:
                    $ref: '#/definitions/ErrorResponse'

    """
    data = request.json

    model = RuPatentsRel()
    result = model.annotate_patents_from_url(data["items"], data["annotate_if_exist"])

    return jsonify(result)


@_server.route('/api/insert/annotation', methods=['POST'])
@exception_handler
def insert_annotations():
    """
        Вставить новую разметку из json в формате labelstudio
        ---
        tags:
            - API
        parameters:
            - name: body
              in: body
              required: true
              schema:
                  type: array
                  items:
                      $ref: '#/definitions/AnnotationRequestItem'
        responses:
            200:
                description: Размеченные данные в формате labelstudio
                schema:
                    type: array
                    items:
                        $ref: '#/definitions/AnnotationResponseItem'

    """
    data = request.json

    model = RuPatentsRel()
    result = model.insert_annotations(data)

    return jsonify(result)


@_server.errorhandler(FileNotFoundError)
def file_not_found_error_handler(error):
    logger.warning('Got error: ' + str(error))
    return str(error), 404


@_server.errorhandler(AssertionError)
def assertion_error(error):
    logger.error(str(error), exc_info=True)
    return str(error), 500


@_server.errorhandler(IndexError)
def index_error(error):
    logger.error(str(error), exc_info=True)
    return str(error), 500


def safe_str_cmp(a, b):
    return hmac.compare_digest(a, b)


@_server.before_request
def log_request_info():
    logger.debug('Request headers: %s', request.headers)
    logger.debug('Request body: %s', request.get_data())


@_server.after_request
def log_response_info(response):
    logger.debug('Response status: %s', response.status)
    logger.debug('Response headers: %s', response.headers)

    if not response.direct_passthrough:
        try:
            logger.debug('Response body: %s', response.get_data())
        except Exception as e:
            logger.warning('Could not log response body: %s', str(e))

    return response


swagger = Swagger(_server, template=swagger_template, parse=True)
