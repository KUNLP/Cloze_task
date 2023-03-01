"""A module with the baseline models for the classification and ranking subtasks."""
from typing import List
import abc

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from scorer import spearmans_rank_correlation


def identity(x):
    return x

from transformers import ElectraModel, ElectraPreTrainedModel
from transformers import RobertaModel, RobertaPreTrainedModel
import torch.nn as nn


class RobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # cls_output = [batch, max_length, hidden_size ] -> last hidden state
        # cls 토큰만 사용할것
        #cls_output_0 = cls_output[:,0,:]
        #logit = self.classifier(cls_output[:, 0, :])
        cls_output = outputs.last_hidden_state
        # cls 토큰만 사용할것
        logit = self.classifier(cls_output[:, 0, :])
        # cls_output = outputs[1]
        # logit = self.classifier(cls_output)
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logit, labels)
            return loss, self.softmax(logit)
        else:
            return self.softmax(logit)


class ElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)

        #cls_output = [batch, max_length, hidden_size ] -> last hidden state
        cls_output = outputs.last_hidden_state
        #cls 토큰만 사용할것
        logit = self.classifier(cls_output[:,0,:])
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logit, labels)
            return loss, self.softmax(logit)
        else:
            return self.softmax(logit)


class Baseline(abc.ABC):
    """An abstract baseline class."""

    def __init__(self):
        self.model = None

    @abc.abstractmethod
    def run_cross_validation(self, instances: List[str], labels: List) -> List:
        """Run k-fold cross-validation on input data.

        :param instances: list of str instances, i. e. sentences with insertions
        :param labels: list of labels
        :return: a list with the performance metric for each cross-validation iteration
        """

    @abc.abstractmethod
    def run_held_out_evaluation(
        self,
        training_instances: List[str],
        training_labels: List,
        dev_instances: List[str],
    ) -> List:
        """Train model on training data and make predictions for development data.

        :param training_instances: list of str training instances, i. e. sentences with insertions
        :param training_labels: list of gold labels for the training instances
        :param dev_instances: list of str development instances, i. e. sentences with insertions
        :return: a list of predictions
        """


class BowClassificationBaseline(Baseline):
    """A baseline for the classification task that combines tf-idf feature extraction and multinomial Naive Bayes."""

    def __init__(self):
        super().__init__()
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
        self.model = Pipeline([("vec", vec), ("cls", MultinomialNB())])

    def run_cross_validation(
        self, instances: List[str], labels: List[int]
    ) -> List[float]:
        """Run k-fold cross-validation on the input data.

        :param instances: list of str instances, i. e. sentences with insertions
        :param labels: list of int gold class indices
        :return: a list with the float accuracy for each run
        """
        instances = [text.split() for text in instances]
        cv = cross_val_score(self.model, instances, labels, cv=5, scoring="accuracy")
        return list(cv)

    def run_held_out_evaluation(
        self,
        training_instances: List[str],
        training_labels: List[int],
        dev_instances: List[str],
    ) -> List[int]:
        """Train model on training data and make predictions for development data.

        :param training_instances: list of str training instances, i. e. sentences with insertions
        :param training_labels: list of int gold class indices for the training instances
        :param dev_instances: list of str development instances, i. e. sentences with insertions
        :return: a list of integers, the class index predictions
        """
        self.model.fit(X=training_instances, y=training_labels)
        return self.model.predict(dev_instances)


class BowRankingBaseline(Baseline):
    """A baseline for the ranking task that combines tf-idf feature extraction and linear regression."""

    def __init__(self):
        super().__init__()
        self.model = Pipeline(
            [("vec", TfidfVectorizer()), ("regr", LinearRegression())]
        )

    def run_cross_validation(
        self, instances: List[str], labels: List[float]
    ) -> List[float]:
        """Run k-fold cross-validation on the input data.

        :param instances: list of str instances, i. e. sentences with insertions
        :param labels: list of float gold ratings
        :return: a list of with the float Spearman's rank correlation coefficient for each run
        """
        scorer = make_scorer(spearmans_rank_correlation, greater_is_better=True)
        scores_per_run = cross_val_score(
            self.model, instances, labels, cv=5, scoring=scorer
        )
        return list(scores_per_run)

    def run_held_out_evaluation(
        self,
        training_instances: List[str],
        training_labels: List[float],
        dev_instances: List[str],
    ) -> List[float]:
        """Train model on training data and make predictions for development data.

        :param training_instances: list of str training instances, i. e. sentences with insertions
        :param training_labels: list of float gold ratings for training instances
        :param dev_instances: list of str development instances, i. e. sentences with insertions
        :return: a list of floats, the class index predictions
        """
        self.model.fit(X=training_instances, y=training_labels)
        return self.model.predict(dev_instances)
