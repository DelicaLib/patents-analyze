from spacy.pipeline.ner import EntityRecognizer
from spacy.language import Language
from thinc.api import Config
from sklearn.metrics import f1_score, precision_recall_fscore_support
import plotly.express as px
import plotly.graph_objects as go
import time
import json
import os
from pathlib import Path


default_model_config = """
[model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = false
nO = null

[model.tok2vec]
@architectures = "spacy-transformers.TransformerListener.v1"
grad_factor = 1.0
pooling = {"@layers":"reduce_mean.v1"}
upstream = "*"
"""
DEFAULT_MODEL = Config().from_str(default_model_config)["model"]


@Language.factory("ner_all_metrics",
    default_config={
        "model": DEFAULT_MODEL,
        "moves": None,
        "scorer": {"@scorers": "spacy.ner_scorer.v1"},
        "incorrect_spans_key": None,
        "update_with_oracle_cut_size": 100,
        "eval_frequency": 100,
    },
    default_score_weights={
        "f1_micro": 1.0,
        "f1_macro": 1.0,
        "f1_weighted": 1.0,
        "f1_COMPONENT": 1.0,
        "f1_SYSTEM": 1.0,
        "f1_ATTRIBUTE": 1.0,
        "ents_p": 0.0,
        "ents_r": 0.0,
    })
def create_ner_all_metrics(
    nlp, name, 
    model, moves, 
    scorer, incorrect_spans_key, 
    update_with_oracle_cut_size, eval_frequency
):
    return NERWithAllMetrics(
        nlp.vocab, model, 
        name=name, moves=moves, 
        scorer=scorer, incorrect_spans_key=incorrect_spans_key,
        update_with_oracle_cut_size=update_with_oracle_cut_size, eval_frequency=eval_frequency
    )


class NERWithAllMetrics(EntityRecognizer):
    
    def __init__(self, *args, eval_frequency=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_history = []
        self.max_f1 = 0
        self.max_f1_step = 0
        self.eval_frequency = eval_frequency
        self.start_learning_time = None
        
    def score(self, examples, **kwargs):
        scores = super().score(examples, **kwargs)
        scores = dict(list(scores.items()) + list(self.custom_scorer(examples).items()))
        tmp_scores = scores.copy()
        tmp_scores["step"] = len(self.metric_history) * self.eval_frequency
        if tmp_scores["f1_macro"] > self.max_f1:
            self.max_f1 = tmp_scores["f1_macro"]
            self.max_f1_step = tmp_scores["step"]
        self.metric_history.append(tmp_scores)
        return scores

    def custom_scorer(self, examples):
      y_true = []
      y_pred = []
      for example in examples:
          gold = {(ent.start_char, ent.end_char, ent.label_) for ent in example.reference.ents}
          pred = {(ent.start_char, ent.end_char, ent.label_) for ent in example.predicted.ents}
          all_spans = gold | pred
          for span in all_spans:
              if span in gold and span in pred:
                  y_true.append(span[2])
                  y_pred.append(span[2])
              elif span in gold:
                  y_true.append(span[2])
                  y_pred.append("O")
              elif span in pred:
                  y_true.append("O")
                  y_pred.append(span[2])

      labels = sorted({label for label in y_true if label != "O"})

      precision, recall, f1, support = precision_recall_fscore_support(
          y_true, y_pred, labels=labels, zero_division=0, average=None
      )
      result = {}
      for l, p, r, f in zip(labels, precision, recall, f1):
          result[f"f1_{l}"] = f

      result["f1_micro"] = f1_score(y_true, y_pred, average="micro", labels=labels, zero_division=0)
      result["f1_macro"] = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
      result["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)

      return result

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
                if trace.name in ["f1_micro", "f1_macro", "f1_weighted"]:
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
