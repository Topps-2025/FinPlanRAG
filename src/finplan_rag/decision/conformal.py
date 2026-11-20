"""Loss-aware nested conformal decision layer for the FinPlan interface.

The layer keeps the original object: a credal state over a finite hypothesis
space, the actions answer / adjudicate / abstain, and an asymmetric decision
loss.  It changes the feasible-set construction from a heuristic credible set
to a split-conformal set, and selects the conformal score, coverage level, and
threshold on calibration/validation data before touching the test year.

The default v3.1 threshold objective is ``realized``: the threshold minimizes
the realized decision loss on the labeled calibration set subject to the
empirical coverage constraint.  The ``expected`` objective, which minimizes
the model's expected policy loss, is retained as an ablation.  The class is
deliberately small and deterministic.  It is the implementation used by
``scripts/run_conformal_decision_v3.py`` and
``scripts/run_deepseek_conformal_set_eval.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .loss import (
    LossConfig,
    action_loss,
    adjudicate_loss,
    answer_loss,
    expected_action_loss,
)

LABELS = ("below", "within", "above")


@dataclass(frozen=True)
class ConformalConfig:
    methods: tuple[str, ...] = ("lac", "aps")
    alphas: tuple[float, ...] = (0.05, 0.10, 0.20)
    n_thresholds: int = 101
    coverage_tolerance: float = 1e-12
    threshold_objectives: tuple[str, ...] = ("realized",)
    threshold_objective: str = "realized"

    def __post_init__(self) -> None:
        for method in self.methods:
            if method not in ("lac", "aps"):
                raise ValueError(f"unknown conformal score: {method}")
        for alpha in self.alphas:
            if not 0.0 < alpha < 1.0:
                raise ValueError(f"alpha must be in (0,1): {alpha}")
        for objective in self.threshold_objectives:
            if objective not in ("expected", "realized"):
                raise ValueError(
                    "threshold objectives must be 'expected' or 'realized'"
                )
        if self.threshold_objective not in ("expected", "realized"):
            raise ValueError(
                "threshold_objective must be 'expected' or 'realized'"
            )


@dataclass(frozen=True)
class ConformalDecision:
    action: str
    prediction: str
    credible_set: tuple[str, ...]
    belief: Mapping[str, float]
    method: str
    alpha: float
    threshold: float


@dataclass
class NestedConformalDecisionLayer:
    config: ConformalConfig = field(default_factory=ConformalConfig)
    method: str | None = None
    alpha: float | None = None
    threshold: float | None = None
    threshold_objective: str | None = None
    calibration_coverage: float | None = None
    validation_loss: float | None = None
    validation_coverage: float | None = None
    fallback: bool = False

    # ------------------------------------------------------------------
    # Conformal sets and scores
    # ------------------------------------------------------------------
    @staticmethod
    def _as_array(probabilities: Sequence[Sequence[float]]) -> np.ndarray:
        array = np.asarray(probabilities, dtype=float)
        if array.ndim != 2 or array.shape[1] != len(LABELS):
            raise ValueError(f"expected (n, {len(LABELS)}) probabilities")
        return array

    @staticmethod
    def lac_sets(probabilities: np.ndarray, threshold: float) -> list[tuple[str, ...]]:
        sets = []
        for row in probabilities:
            chosen = tuple(
                label
                for index, label in enumerate(LABELS)
                if row[index] >= 1.0 - threshold
            )
            if not chosen:
                chosen = (LABELS[int(np.argmax(row))],)
            sets.append(chosen)
        return sets

    @staticmethod
    def aps_sets(probabilities: np.ndarray, threshold: float) -> list[tuple[str, ...]]:
        sets = []
        for row in probabilities:
            order = np.argsort(-row, kind="stable")
            chosen = []
            mass = 0.0
            for index in order:
                chosen.append(LABELS[index])
                mass += float(row[index])
                if mass >= threshold:
                    break
            sets.append(tuple(chosen))
        return sets

    @classmethod
    def sets(
        cls, probabilities: Sequence[Sequence[float]], threshold: float, method: str
    ) -> list[tuple[str, ...]]:
        array = cls._as_array(probabilities)
        if method == "lac":
            return cls.lac_sets(array, threshold)
        if method == "aps":
            return cls.aps_sets(array, threshold)
        raise ValueError(method)

    @staticmethod
    def scores(
        probabilities: Sequence[Sequence[float]], truths: Sequence[str], method: str
    ) -> np.ndarray:
        array = np.asarray(probabilities, dtype=float)
        scores = []
        for row, truth in zip(array, truths):
            index = LABELS.index(str(truth))
            if method == "lac":
                scores.append(1.0 - float(row[index]))
            elif method == "aps":
                order = np.argsort(-row, kind="stable")
                mass = 0.0
                for position in order:
                    mass += float(row[position])
                    if LABELS[position] == str(truth):
                        break
                scores.append(mass)
            else:
                raise ValueError(method)
        return np.asarray(scores, dtype=float)

    # ------------------------------------------------------------------
    # Loss-aware action selection
    # ------------------------------------------------------------------
    @staticmethod
    def _belief(row: np.ndarray) -> dict[str, float]:
        return {label: float(value) for label, value in zip(LABELS, row)}

    @classmethod
    def expected_losses(
        cls,
        row: Sequence[float],
        candidate_set: Sequence[str],
        loss: LossConfig,
    ) -> dict[str, float]:
        array = np.asarray(row, dtype=float)
        belief = cls._belief(array)
        prediction = LABELS[int(np.argmax(array))]
        return {
            "answer": expected_action_loss(
                "answer", belief, loss, prediction=prediction
            ),
            "adjudicate": expected_action_loss(
                "adjudicate", belief, loss, credible_set=tuple(candidate_set)
            ),
            "abstain": loss.abstain,
        }

    @classmethod
    def choose_action(
        cls,
        row: Sequence[float],
        candidate_set: Sequence[str],
        loss: LossConfig,
    ) -> tuple[str, str]:
        array = np.asarray(row, dtype=float)
        prediction = LABELS[int(np.argmax(array))]
        values = cls.expected_losses(array, candidate_set, loss)
        action = min(("answer", "adjudicate", "abstain"), key=lambda key: values[key])
        return action, prediction

    @classmethod
    def policy_loss(
        cls,
        probabilities: Sequence[Sequence[float]],
        sets: Sequence[Sequence[str]],
        loss: LossConfig,
    ) -> float:
        values = []
        for row, candidate_set in zip(probabilities, sets):
            action, prediction = cls.choose_action(row, candidate_set, loss)
            values.append(
                cls.expected_losses(row, candidate_set, loss)[action]
            )
        return float(np.mean(values))

    @classmethod
    def realized_policy_loss(
        cls,
        probabilities: Sequence[Sequence[float]],
        sets: Sequence[Sequence[str]],
        truths: Sequence[str],
        loss: LossConfig,
    ) -> float:
        """Realized decision loss of the policy on labeled calibration data.

        The action is still chosen without the label; the label is used only
        to score the threshold.  This is the objective used by the v3.1
        layer: it avoids selecting a threshold on a miscalibrated expected
        loss.
        """
        values = []
        for row, candidate_set, truth in zip(probabilities, sets, truths):
            action, prediction = cls.choose_action(row, candidate_set, loss)
            values.append(
                action_loss(
                    action,
                    truth,
                    loss,
                    prediction=prediction,
                    credible_set=tuple(candidate_set),
                )
            )
        return float(np.mean(values))

    # ------------------------------------------------------------------
    # Threshold and method selection
    # ------------------------------------------------------------------
    def select_threshold(
        self,
        probabilities: Sequence[Sequence[float]],
        truths: Sequence[str],
        loss: LossConfig,
        method: str,
        alpha: float,
        objective: str | None = None,
    ) -> tuple[float, float, float]:
        array = self._as_array(probabilities)
        objective = objective or self.config.threshold_objective
        if objective not in ("expected", "realized"):
            raise ValueError(objective)
        scores = self.scores(array, truths, method)
        candidates = np.unique(
            np.concatenate(
                (
                    [0.0],
                    np.quantile(scores, np.linspace(0.0, 1.0, self.config.n_thresholds)),
                    [1.0],
                )
            )
        )
        best: tuple[float, float] | None = None
        best_coverage = 1.0
        for threshold in candidates:
            candidate_sets = self.sets(array, float(threshold), method)
            coverage = float(
                np.mean(
                    [
                        truth in candidate
                        for truth, candidate in zip(truths, candidate_sets)
                    ]
                )
            )
            if coverage + self.config.coverage_tolerance < 1.0 - alpha:
                continue
            if objective == "realized":
                value = self.realized_policy_loss(
                    array, candidate_sets, truths, loss
                )
            else:
                value = self.policy_loss(array, candidate_sets, loss)
            key = (value, float(threshold))
            if best is None or key < best:
                best = key
                best_coverage = coverage
        if best is None:
            threshold = 1.0
            candidate_sets = self.sets(array, threshold, method)
            coverage = float(
                np.mean(
                    [
                        truth in candidate
                        for truth, candidate in zip(truths, candidate_sets)
                    ]
                )
            )
            value = (
                self.realized_policy_loss(array, candidate_sets, truths, loss)
                if objective == "realized"
                else self.policy_loss(array, candidate_sets, loss)
            )
            return threshold, value, coverage
        return best[1], best[0], best_coverage

    def fit(
        self,
        calibration_probabilities: Sequence[Sequence[float]],
        calibration_truths: Sequence[str],
        validation_probabilities: Sequence[Sequence[float]],
        validation_truths: Sequence[str],
        loss: LossConfig,
    ) -> "NestedConformalDecisionLayer":
        candidates = []
        for objective in self.config.threshold_objectives:
            for method in self.config.methods:
                for alpha in self.config.alphas:
                    threshold, calibration_loss, calibration_coverage = self.select_threshold(
                        calibration_probabilities,
                        calibration_truths,
                        loss,
                        method,
                        alpha,
                        objective=objective,
                    )
                    validation_sets = self.sets(
                        validation_probabilities, threshold, method
                    )
                    validation = self.evaluate(
                        validation_probabilities, validation_truths, validation_sets, loss
                    )
                    candidates.append(
                        {
                            "objective": objective,
                            "method": method,
                            "alpha": alpha,
                            "threshold": threshold,
                            "calibration_loss": calibration_loss,
                            "calibration_coverage": calibration_coverage,
                            "validation_loss": validation["mean_loss"],
                            "validation_coverage": validation["coverage"],
                        }
                    )
        if not candidates:
            raise ValueError("no conformal candidate was produced")
        valid = [
            item
            for item in candidates
            if item["validation_coverage"] + self.config.coverage_tolerance
            >= 1.0 - item["alpha"]
        ]
        self.fallback = not bool(valid)
        if valid:
            chosen = min(
                valid,
                key=lambda item: (item["validation_loss"], item["alpha"]),
            )
        else:
            # Fallback: the most conservative validation coverage; this is
            # explicit and is reported as a violated coverage constraint.
            chosen = max(
                candidates,
                key=lambda item: (
                    item["validation_coverage"],
                    -item["validation_loss"],
                ),
            )
        self.threshold_objective = chosen["objective"]
        self.method = chosen["method"]
        self.alpha = chosen["alpha"]
        self.threshold = chosen["threshold"]
        self.calibration_coverage = chosen["calibration_coverage"]
        self.validation_loss = chosen["validation_loss"]
        self.validation_coverage = chosen["validation_coverage"]
        return self

    def recalibrate(
        self,
        calibration_probabilities: Sequence[Sequence[float]],
        calibration_truths: Sequence[str],
        loss: LossConfig,
    ) -> "NestedConformalDecisionLayer":
        if self.method is None or self.alpha is None:
            raise RuntimeError("fit must be called before recalibrate")
        threshold, _, coverage = self.select_threshold(
            calibration_probabilities,
            calibration_truths,
            loss,
            self.method,
            self.alpha,
            objective=self.threshold_objective,
        )
        self.threshold = threshold
        self.calibration_coverage = coverage
        return self

    # ------------------------------------------------------------------
    # Prediction and evaluation
    # ------------------------------------------------------------------
    def predict(
        self,
        probabilities: Sequence[Sequence[float]],
        loss: LossConfig,
    ) -> list[ConformalDecision]:
        if self.method is None or self.alpha is None or self.threshold is None:
            raise RuntimeError("fit must be called before predict")
        array = self._as_array(probabilities)
        candidate_sets = self.sets(array, self.threshold, self.method)
        decisions = []
        for row, candidate_set in zip(array, candidate_sets):
            belief = self._belief(row)
            action, prediction = self.choose_action(row, candidate_set, loss)
            decisions.append(
                ConformalDecision(
                    action=action,
                    prediction=prediction,
                    credible_set=tuple(candidate_set),
                    belief=belief,
                    method=self.method,
                    alpha=self.alpha,
                    threshold=self.threshold,
                )
            )
        return decisions

    def evaluate(
        self,
        probabilities: Sequence[Sequence[float]],
        truths: Sequence[str],
        candidate_sets: Sequence[Sequence[str]],
        loss: LossConfig,
    ) -> dict[str, float | int]:
        losses = []
        covered = 0
        sizes = []
        actions = {"answer": 0, "adjudicate": 0, "abstain": 0}
        array = self._as_array(probabilities)
        for row, truth, candidate_set in zip(array, truths, candidate_sets):
            action, prediction = self.choose_action(row, candidate_set, loss)
            actions[action] += 1
            losses.append(
                action_loss(
                    action,
                    truth,
                    loss,
                    prediction=prediction,
                    credible_set=tuple(candidate_set),
                )
            )
            covered += int(truth in candidate_set)
            sizes.append(len(candidate_set))
        n = len(truths)
        return {
            "n": n,
            "mean_loss": float(np.mean(losses)),
            "coverage": covered / n,
            "mean_set_size": float(np.mean(sizes)),
            "answer_rate": actions["answer"] / n,
            "adjudicate_rate": actions["adjudicate"] / n,
            "abstain_rate": actions["abstain"] / n,
        }
