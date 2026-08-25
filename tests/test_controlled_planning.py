import inspect
import math

from finplan_rag import controlled_planning as bench


def test_non_oracle_planners_do_not_read_hidden_truth():
    source = inspect.getsource(bench.infer) + inspect.getsource(bench.finplan_target)
    assert "hidden_paths" not in source
    assert "truth_state" not in source


def test_all_methods_share_budget_and_return_valid_states():
    case = bench.make_case("balanced", 0, 0)
    for method in bench.METHODS:
        pred = bench.infer(case, method, budget=7)
        assert pred.state in set(bench.STATES) | {"unknown"}
        assert pred.queries <= 7


def test_future_documents_are_filtered_except_no_time_ablation():
    case = bench.make_case("future_revision", 1, 3)
    for method in set(bench.METHODS) - {"finplan_no_time"}:
        pred = bench.infer(case, method, budget=10)
        # Retrieval may spend cost on a future hit, but the resolver must not
        # consume it. This invariant is tested indirectly by recomputing from
        # only point-in-time documents.
        observed = [d for d in case.documents if d.doc_id in pred.used_doc_ids and d.available_at <= case.cutoff]
        state, closed = bench.aggregate_prediction(
            case,
            observed,
            ignore_counterevidence=method == "finplan_no_counterevidence",
        )
        assert (state, closed) == (pred.state, pred.closed)


def test_no_time_ablation_keeps_the_same_adaptive_stopping_rule():
    slot_values = {
        "entity": "satisfied",
        "concept": "satisfied",
        "status": "realized",
        "counterevidence": "clear",
    }
    documents = tuple(
        bench.Document(i, "p0", "product", slot, value, "official", 5, 0.9, True, False)
        for i, (slot, value) in enumerate(slot_values.items())
    )
    path = bench.PathTruth("p0", "product", "direct", slot_values)
    case = bench.Case("stop-check", "balanced", 10, (("p0", "product"),), documents, (path,), "direct")

    full = bench.infer(case, "finplan_state", budget=10)
    no_time = bench.infer(case, "finplan_no_time", budget=10)

    assert full.closed and no_time.closed
    assert full.queries == no_time.queries < 10


def test_protocol_outputs_finite_primary_metrics():
    result = bench.run(seeds=2, cases_per_world=8, budget=6)
    for method in bench.METHODS:
        for metric in ("accuracy", "coverage", "overclaim", "closure", "queries", "utility"):
            assert math.isfinite(result["summary"][method][metric])
