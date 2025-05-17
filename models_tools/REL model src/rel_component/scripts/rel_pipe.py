import json
import os
import time
from itertools import islice
from pathlib import Path
from typing import Tuple, List, Iterable, Optional, Dict, Callable, Any

from sklearn.metrics import precision_recall_fscore_support, f1_score
from thinc.types import Floats2d
import numpy
from spacy.training.example import Example
from thinc.api import Model, Optimizer
from spacy.tokens.doc import Doc
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.vocab import Vocab
from spacy import Language
from thinc.model import set_dropout_rate
from wasabi import Printer
import plotly.express as px
import plotly.graph_objects as go

Doc.set_extension("rel", default={}, force=True)
msg = Printer()


@Language.factory(
    "relation_extractor",
    requires=["doc.ents", "token.ent_iob", "token.ent_type"],
    assigns=["doc._.rel"],
    default_score_weights={
        "rel_micro_p": 0.0,
        "rel_micro_r": 0.0,
        "rel_micro_f": 1.0,
        "rel_macro_f": 1.0,
        "rel_weighted_f": 1.0,
        "f1_PART-OF": 1.0,
        "f1_LOCATED-AT": 1.0,
        "f1_CONNECTED-WITH": 1.0,
        "f1_IN-MANNER-OF": 1.0,
        "f1_ATTRIBUTE-FOR": 1.0
    },
)
def make_relation_extractor(
    nlp: Language, name: str, model: Model, eval_frequency, *, threshold: float
):
    """Construct a RelationExtractor component."""
    return RelationExtractor(nlp.vocab, model, name, threshold=threshold, eval_frequency=eval_frequency)


class RelationExtractor(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "rel",
        *,
        threshold: float,
        eval_frequency = 100
    ) -> None:
        """Initialize a relation extractor."""
        self.vocab = vocab
        self.model = model
        self.name = name
        self.cfg = {"labels": [], "threshold": threshold}
        self.eval_frequency = eval_frequency
        self.start_learning_time = None
        self.metric_history = []
        self.max_f1 = 0
        self.max_f1_step = 0

    @property
    def labels(self) -> Tuple[str]:
        """Returns the labels currently added to the component."""
        return tuple(self.cfg["labels"])

    @property
    def threshold(self) -> float:
        """Returns the threshold above which a prediction is seen as 'True'."""
        return self.cfg["threshold"]

    def add_label(self, label: str) -> int:
        """Add a new label to the pipe."""
        if not isinstance(label, str):
            raise ValueError("Only strings can be added as labels to the RelationExtractor")
        if label in self.labels:
            return 0
        self.cfg["labels"] = list(self.labels) + [label]
        return 1

    def __call__(self, doc: Doc) -> Doc:
        """Apply the pipe to a Doc."""
        # check that there are actually any candidate instances in this batch of examples
        total_instances = len(self.model.attrs["get_instances"](doc))
        if total_instances == 0:
            msg.info("Could not determine any instances in doc - returning doc as is.")
            return doc

        predictions = self.predict([doc])
        self.set_annotations([doc], predictions)
        return doc

    def predict(self, docs: Iterable[Doc]) -> Floats2d:
        """Apply the pipeline's model to a batch of docs, without modifying them."""
        get_instances = self.model.attrs["get_instances"]
        total_instances = sum([len(get_instances(doc)) for doc in docs])
        if total_instances == 0:
            msg.info("Could not determine any instances in any docs - can not make any predictions.")
        scores = self.model.predict(docs)
        return self.model.ops.asarray(scores)

    def set_annotations(self, docs: Iterable[Doc], scores: Floats2d) -> None:
        """Modify a batch of `Doc` objects, using pre-computed scores."""
        c = 0
        get_instances = self.model.attrs["get_instances"]
        for doc in docs:
            for (e1, e2) in get_instances(doc):
                offset = (e1.start, e2.start)
                if offset not in doc._.rel:
                    doc._.rel[offset] = {}
                for j, label in enumerate(self.labels):
                    doc._.rel[offset][label] = scores[c, j]
                c += 1

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        set_annotations: bool = False,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Learn from a batch of documents and gold-standard information,
        updating the pipe's model. Delegates to predict and get_loss."""
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        set_dropout_rate(self.model, drop)

        # check that there are actually any candidate instances in this batch of examples
        total_instances = 0
        for eg in examples:
            total_instances += len(self.model.attrs["get_instances"](eg.predicted))
        if total_instances == 0:
            msg.info("Could not determine any instances in doc.")
            return losses

        # run the model
        docs = [eg.predicted for eg in examples]
        predictions, backprop = self.model.begin_update(docs)
        loss, gradient = self.get_loss(examples, predictions)
        backprop(gradient)
        if sgd is not None:
            self.model.finish_update(sgd)
        losses[self.name] += loss
        if set_annotations:
            self.set_annotations(docs, predictions)
        return losses

    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, float]:
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores."""
        truths = self._examples_to_truth(examples)
        gradient = scores - truths
        mean_square_error = (gradient ** 2).sum(axis=1).mean()
        return float(mean_square_error), gradient

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Language = None,
        labels: Optional[List[str]] = None,
    ):
        """Initialize the pipe for training, using a representative set
        of data examples.
        """
        if labels is not None:
            for label in labels:
                self.add_label(label)
        else:
            for example in get_examples():
                relations = example.reference._.rel
                for indices, label_dict in relations.items():
                    for label in label_dict.keys():
                        self.add_label(label)
        self._require_labels()

        subbatch = list(islice(get_examples(), 10))
        doc_sample = [eg.reference for eg in subbatch]
        label_sample = self._examples_to_truth(subbatch)
        if label_sample is None:
            raise ValueError("Call begin_training with relevant entities and relations annotated in "
                             "at least a few reference examples!")
        self.model.initialize(X=doc_sample, Y=label_sample)

    def _examples_to_truth(self, examples: List[Example]) -> Optional[numpy.ndarray]:
        # check that there are actually any candidate instances in this batch of examples
        nr_instances = 0
        for eg in examples:
            nr_instances += len(self.model.attrs["get_instances"](eg.reference))
        if nr_instances == 0:
            return None

        truths = numpy.zeros((nr_instances, len(self.labels)), dtype="f")
        c = 0
        for i, eg in enumerate(examples):
            for (e1, e2) in self.model.attrs["get_instances"](eg.reference):
                gold_label_dict = eg.reference._.rel.get((e1.start, e2.start), {})
                for j, label in enumerate(self.labels):
                    truths[c, j] = gold_label_dict.get(label, 0)
                c += 1

        truths = self.model.ops.asarray(truths)
        return truths

    def score(self, examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
        """Score a batch of examples."""
        scores = score_relations(examples, self.threshold)

        tmp_scores = scores.copy()
        tmp_scores["step"] = len(self.metric_history) * self.eval_frequency
        if tmp_scores["rel_macro_f"] > self.max_f1:
            self.max_f1 = tmp_scores["rel_macro_f"]
            self.max_f1_step = tmp_scores["step"]
        self.metric_history.append(tmp_scores)

        return scores

    def preprocess_metric_history(self):
        result = {
            "metric_name": [],
            "metric_value": [],
            "step": []
        }
        for cur_metrics in self.metric_history:
            cur_step = cur_metrics["step"]
            for key, value in cur_metrics.items():
                if key != "step" and isinstance(value, float):
                    result["metric_name"].append(key)
                    result["metric_value"].append(value)
                    result["step"].append(cur_step)
        return result

    def save_metrics_history(self, path):
        if self.start_learning_time is None:
            self.start_learning_time = time.monotonic()

        if self.metric_history:

            metrics_history_to_save = self.preprocess_metric_history()
            fig = px.line(metrics_history_to_save, x="step", y="metric_value", color="metric_name")
            for trace in fig.data:
                if trace.name in ["rel_micro_f", "rel_macro_f", "rel_weighted_f"]:
                    trace.line.width = 6
                else:
                    trace.line.width = 1

                idx = list(trace.x).index(self.max_f1_step)
                highlight_y = list(trace.y)[idx]
                line_color = trace.line.color
                line_name = trace.name
                fig.add_trace(go.Scatter(
                    x=[self.max_f1_step], y=[highlight_y],
                    mode='markers+text',
                    marker=dict(
                        color=line_color, size=10),
                        text=[f"{round(highlight_y, 2)}"],
                        textposition="top center",
                        name=f"{line_name} best"
                    ))

            current_time = time.monotonic()
            current_time_of_training = current_time - self.start_learning_time
            current_time_of_training_text = f"{int(current_time_of_training // 3600)} hrs {int(current_time_of_training % 3600) // 60} min {round(current_time_of_training % 60)} sec"

            fig.update_layout(title = dict(
                text="Training statistics",
                subtitle=dict(
                    text=f"Training time amounted to {current_time_of_training_text}",
                    font=dict(color="gray", size=13),
                )
            ))

            output_dir = os.path.join(str(path), "logs")
            os.makedirs(output_dir, exist_ok=True)
            fig_path = os.path.join(output_dir, "training_metrics.html")
            json_path = os.path.join(output_dir, "training_metrics.json")
            fig.write_html(fig_path)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "data": metrics_history_to_save,
                    "train_time_s": current_time_of_training
                }, f, indent=2, ensure_ascii=False)

    def to_disk(self, path, *args, **kwargs):
        super().to_disk(path, *args, **kwargs)
        output_dir = Path(path)
        output_dir_metrics = output_dir.parent.parent
        self.save_metrics_history(output_dir_metrics)


def score_relations(examples: Iterable[Example], threshold: float) -> Dict[str, Any]:
    """Score a batch of examples."""

    y_true = []
    y_pred = []
    for example in examples:
        gold = example.reference._.rel
        pred = example.predicted._.rel
        for key, pred_dict in pred.items():
            gold_labels = {k for (k, v) in gold.get(key, {}).items() if v == 1.0}
            for k, v in pred_dict.items():
                if v >= threshold:
                    if k in gold_labels:
                        y_true.append(k)
                        y_pred.append(k)
                    else:
                        y_true.append("O")
                        y_pred.append(k)
                else:
                    if k in gold_labels:
                        y_true.append(k)
                        y_pred.append("O")


    labels = sorted({label for label in y_true if label != "O"})

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0, average=None
    )
    result = {}
    for l, p, r, f in zip(labels, precision, recall, f1):
        result[f"f1_{l}"] = f

    p, r, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0, average="micro", beta=1
    )

    result["rel_micro_p"] = p
    result["rel_micro_r"] = r
    result["rel_micro_f"] = f1_micro
    result["rel_macro_f"] = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    result["rel_weighted_f"] = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)

    return result
