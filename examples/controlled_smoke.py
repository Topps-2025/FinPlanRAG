"""Run a small, deterministic FinPlan-RAG planning diagnostic."""

from finplan_rag.controlled_planning import run


if __name__ == "__main__":
    result = run(seeds=2, cases_per_world=8, budget=6)
    for method, metrics in result["summary"].items():
        print(f"{method}: closure={metrics['closure']:.3f}, queries={metrics['queries']:.2f}")
