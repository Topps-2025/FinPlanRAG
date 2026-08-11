"""Fill the LOFin full-run results into the docs 5.13 section (Task #17).

Reads results/lofin_full_benchmark_audit_v1.json (the audit's final output)
and fills the marker-delimited regions in
docs/04-数据与实验/06-LOFin多文档轨道实验.md:
  <!-- RESULTS_FULL_TABLE_BEGIN/END -->  -> per-stratum x per-method metric tables
  <!-- RESULTS_P_CHECKS_BEGIN/END -->    -> the audit's P1-P6 check lines + McNemar tally
Idempotent: re-running replaces the region between the markers.

Usage: python render_lofin_docs_blocks.py [audit.json] [docs.md]
Pure mechanical step; all numbers come from the audit JSON (no re-derivation).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

AUDIT = Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\results\lofin_full_benchmark_audit_v1.json")
DOCS = Path(r"C:\Users\Lenovo\Desktop\Paper\FinPlanRAG\docs\04-数据与实验\06-LOFin多文档轨道实验.md")

METHODS = ["finplan_v9_cascade", "metadata_v9_fill", "finplan_v8", "finplan_v7",
           "finplan_v4", "single_shot", "period_metadata_decomposition",
           "generic_adaptive_period", "hirec_period"]
METHODS_ZH = {"finplan_v9_cascade": "v9_cascade", "metadata_v9_fill": "v9+meta_fill",
              "finplan_v8": "v8", "finplan_v7": "v7", "finplan_v4": "v4",
              "single_shot": "single_shot", "period_metadata_decomposition": "period_meta",
              "generic_adaptive_period": "generic_adapt", "hirec_period": "hirec_period"}
STRATA = ["full", "textual", "numeric_table", "numeric_text", "single_doc",
          "multi_doc", "names_company", "no_company_name", "fiscal_year_phrasing"]
STRATA_ZH = {"full": "全集", "textual": "textual(文本)", "numeric_table": "numeric_table(数值表格)",
             "numeric_text": "numeric_text(数值文本)", "single_doc": "单证据",
             "multi_doc": "多证据", "names_company": "题面命名公司",
             "no_company_name": "题面未命名公司",
             "fiscal_year_phrasing": "fiscal year N / FY N 措辞"}


def metric_table(title: str, key: str, fmt: str) -> list[str]:
    ps = audit["per_stratum"]
    header = [f"**{title}**", "",
              "| 分层 | " + " | ".join(METHODS_ZH[m] for m in METHODS) + " |",
              "| --- | " + " | ".join("---" for _ in METHODS) + " |"]
    body = []
    for s in STRATA:
        cells = []
        for m in METHODS:
            v = ps.get(s, {}).get(m)
            val = v[key] if v else None
            cells.append(fmt.format(val) if val is not None else "-")
        body.append(f"| {STRATA_ZH[s]} | " + " | ".join(cells) + " |")
    return header + body


def main() -> None:
    audit_path = Path(sys.argv[1]) if len(sys.argv) > 1 else AUDIT
    docs_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DOCS
    global audit
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    table_blocks = []
    for title, key, fmt in (("闭包 (cascade_closure / closure)", "closure", "{:.3f}"),
                            ("错误文档率 (wrong_doc_rate)", "wrong_doc_rate", "{:.2f}"),
                            ("义务精确匹配 (obligation_exact, 仅 planner 系)", "obligation_exact",
                             "{:.2f}"),
                            ("未来泄漏 case 数 (future_leak)", "future_leak", "{}")):
        table_blocks += metric_table(title, key, fmt) + [""]
    table_blocks = table_blocks[:-1]  # drop the trailing blank line

    sig = [p for p in audit.get("mcnemar_pairs", []) if p["exact_p"] < 0.05]
    p_blocks = (["**预注册预测 P1–P6 检验**"] + audit.get("p_checks", []) +
                [f"- 36 对两两精确 McNemar：{len(sig)} 对在 p<0.05（未做多重比较校正）显著；"
                 f"显著对与完整 36 对表见 audit 输出 / results/lofin_full_benchmark_audit_v1.json"])

    text = docs_path.read_text(encoding="utf-8")

    def replace_region(begin: str, end: str, blk: list[str]) -> str:
        """Marker-delimited regions make re-rendering idempotent: a chain
        interruption after a successful render cannot abort the next firing
        (a single-consumed placeholder would have raised 'not found')."""
        content = "\n".join(blk)
        if begin in text and end in text:
            start = text.index(begin)
            stop = text.index(end) + len(end)
            return text[:start] + begin + "\n" + content + "\n" + end + text[stop:]
        raise SystemExit(f"marker region not found in {docs_path}: {begin}")

    text = replace_region("<!-- RESULTS_FULL_TABLE_BEGIN -->",
                          "<!-- RESULTS_FULL_TABLE_END -->", table_blocks)
    text = replace_region("<!-- RESULTS_P_CHECKS_BEGIN -->",
                          "<!-- RESULTS_P_CHECKS_END -->", p_blocks)
    docs_path.write_text(text, encoding="utf-8")
    print(f"filled placeholders in {docs_path}")


if __name__ == "__main__":
    main()
