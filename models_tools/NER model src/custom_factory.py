from spacy.pipeline.ner import EntityRecognizer
from spacy.language import DEFAULT_CONFIG, Language
from thinc.api import Config
from sklearn.metrics import f1_score, precision_recall_fscore_support


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
    "update_with_oracle_cut_size": 100
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
def create_ner_all_metrics(nlp, name, model, moves, scorer, incorrect_spans_key, update_with_oracle_cut_size):
    return NERWithAllMetrics(nlp.vocab, model, name=name, moves=moves, scorer=scorer,
                              incorrect_spans_key=incorrect_spans_key,
                              update_with_oracle_cut_size=update_with_oracle_cut_size)

class NERWithAllMetrics(EntityRecognizer):
    def score(self, examples, **kwargs):
        scores = super().score(examples, **kwargs)
        scores = dict(list(scores.items()) + list(self.custom_scorer(examples).items()))
        del scores["ents_f"]
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
