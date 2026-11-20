"""Prepare and run the cloud-GPU experiment suite.

Three backbones are used:

* DeepSeek Flash through the Anthropic-compatible API;
* GLM 5.3 Flash through the Zhipu API;
* Qwen3-8B served locally by vLLM on a 32 GB GPU.

The script generates per-backbone configs from the existing experiment configs,
then runs anchor extraction, fusion, named baselines, official-source
baselines, qualification ablation and the A-share backtest.  It never stores an
API key.  By default it is a dry run: it writes the exact commands to
``paper/results/cloud/commands.sh`` without calling a model.

Usage:
    python scripts/run_cloud_suite.py --dry-run
    python scripts/run_cloud_suite.py --execute --stages anchors,fusion
    python scripts/run_cloud_suite.py --execute --backbones qwen3 --stages named,official
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKBONES = REPO_ROOT / "configs" / "cloud" / "backbones.yaml"
BASE_EVOLUTION = REPO_ROOT / "experiments" / "evolution" / "config.yaml"
BASE_ANCHOR = REPO_ROOT / "configs" / "anchor_cli.yaml"
CLOUD_ROOT = REPO_ROOT / "paper" / "results" / "cloud"
ALL_STAGES = ("prepare", "anchors", "rf", "fusion", "named", "official", "qualification", "ashare")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def backbone_configs(name: str, spec: dict[str, Any]) -> tuple[Path, Path]:
    """Create per-backbone evolution and anchor-CLI configs."""
    cloud_dir = CLOUD_ROOT / name
    cloud_dir.mkdir(parents=True, exist_ok=True)

    evolution = load_yaml(BASE_EVOLUTION)
    evolution["results_dir"] = str(cloud_dir / "evolution")
    evolution["sources"] = dict(evolution.get("sources") or {})
    evolution["sources"]["validation_new"] = str(cloud_dir / "evolution" / "anchors_validation.jsonl")
    evolution["sources"]["test_new"] = str(cloud_dir / "evolution" / "anchors_test.jsonl")
    evolution["llm"] = {
        "provider": spec["provider"],
        "model": spec["model"],
        "base_url": spec["base_url"],
        "api_key_env": spec["api_key_env"],
        "workers": int(spec.get("workers") or 8),
        "max_tokens": int(spec.get("max_tokens") or 1800),
        "temperature": float(spec.get("temperature") or 0.0),
        "cache_dir": str(cloud_dir / "llm_cache"),
    }
    if spec.get("reasoning_effort"):
        evolution["llm"]["reasoning_effort"] = spec["reasoning_effort"]
    if spec.get("extra_body"):
        evolution["llm"]["extra_body"] = spec["extra_body"]
    evolution_path = CLOUD_ROOT / "configs" / f"{name}_evolution.yaml"
    write_yaml(evolution_path, evolution)

    anchor = load_yaml(BASE_ANCHOR)
    anchor["llm"] = {
        "provider": spec["provider"],
        "model": spec["model"],
        "base_url": spec["base_url"],
        "api_key_env": spec["api_key_env"],
        "max_tokens": 3200,
        "temperature": 0.0,
        "cache_dir": str(cloud_dir / "anchor_cache"),
    }
    if spec.get("extra_body"):
        anchor["llm"]["extra_body"] = spec["extra_body"]
    anchor_path = CLOUD_ROOT / "configs" / f"{name}_anchor_cli.yaml"
    write_yaml(anchor_path, anchor)
    return evolution_path, anchor_path


def command_env(name: str, spec: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts"), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    env["PYTHONUTF8"] = "1"
    if name == "qwen3" and not env.get("QWEN_API_KEY"):
        env["QWEN_API_KEY"] = "dummy"
    return env


def check_backbone(name: str, spec: dict[str, Any]) -> list[str]:
    problems = []
    if not os.environ.get(str(spec["api_key_env"])):
        if name != "qwen3":
            problems.append(f"{name}: missing {spec['api_key_env']}")
    if name == "qwen3":
        base = str(spec["base_url"]).rstrip("/")
        try:
            import urllib.request
            with urllib.request.urlopen(base + "/models", timeout=3) as response:
                if response.status >= 400:
                    problems.append(f"qwen3: vLLM returned HTTP {response.status}")
        except Exception as error:  # noqa: BLE001
            problems.append(f"qwen3: vLLM is not reachable at {base}: {error}")
    return problems


def command(argv: list[str]) -> str:
    return shlex.join(argv)


def build_commands(
    selected: list[str],
    specs: dict[str, dict[str, Any]],
    stages: tuple[str, ...],
    limit: int,
) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    py = sys.executable
    common_limit = ["--limit", str(limit)] if limit else []

    for name in selected:
        spec = specs[name]
        evolution_path, anchor_path = backbone_configs(name, spec)
        cloud_dir = CLOUD_ROOT / name
        if "anchors" in stages:
            for split in ("validation", "test"):
                commands.append((
                    f"{name}:anchors:{split}",
                    [py, "experiments/evolution/run_anchors.py", "--config", str(evolution_path),
                     "--split", split, *common_limit],
                ))
        if "fusion" in stages:
            commands.append((
                f"{name}:fusion",
                [py, "experiments/evolution/temporal_fusion.py", "--config", str(evolution_path),
                 "--rf-validation", "paper/results/evolution_v1/rf_validation.jsonl",
                 "--rf-test", "paper/results/rf_calibrated_repro/per_claim.jsonl",
                 "--tag", "cloud"],
            ))
        extra_body = json.dumps(spec.get("extra_body") or {}, ensure_ascii=False)
        if "named" in stages:
            commands.append((
                f"{name}:named",
                [py, "scripts/run_named_baselines.py", "--data", "paper/data/dataset_v4.json",
                 "--out", str(cloud_dir / "named"), "--verified-only", "--workers", str(spec.get("workers") or 8),
                 "--provider", str(spec["provider"]), "--model", str(spec["model"]),
                 "--base-url", str(spec["base_url"]), "--api-key-env", str(spec["api_key_env"]),
                 "--extra-body", extra_body, *common_limit],
            ))
        if "official" in stages:
            commands.append((
                f"{name}:official",
                [py, "scripts/run_official_baselines.py", "--data", "paper/data/dataset_v4.json",
                 "--out", str(cloud_dir / "official"), "--verified-only", "--workers", str(spec.get("workers") or 8),
                 "--provider", str(spec["provider"]), "--model", str(spec["model"]),
                 "--base-url", str(spec["base_url"]), "--api-key-env", str(spec["api_key_env"]),
                 "--extra-body", extra_body, *common_limit],
            ))
        if "qualification" in stages:
            commands.append((
                f"{name}:qualification",
                [py, "scripts/run_qualification_ablation.py", "--config", str(evolution_path),
                 "--out", str(cloud_dir / "qualification")],
            ))
        if "ashare" in stages:
            commands.append((
                f"{name}:ashare",
                [py, "scripts/run_ashare_backtest.py", "--data", "paper/data/ashare_backtest/claims.json",
                 "--out", str(cloud_dir / "ashare"), "--config", str(anchor_path),
                 "--methods", "full,no_bridge", *common_limit],
            ))

    if "rf" in stages:
        commands.append((
            "rf",
            [py, "scripts/run_company_disjoint_benchmark.py", "--data", "paper/data/dataset_v4.json",
             "--verified-only", "--out", str(CLOUD_ROOT / "rf")],
        ))
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", default=",".join(["deepseek", "glm", "qwen3"]))
    parser.add_argument("--stages", default=",".join(ALL_STAGES))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = load_yaml(DEFAULT_BACKBONES)
    specs = payload["backbones"]
    selected = [value.strip() for value in args.backbones.split(",") if value.strip()]
    stages = tuple(value.strip() for value in args.stages.split(",") if value.strip())
    for name in selected:
        if name not in specs:
            raise SystemExit(f"unknown backbone: {name}")
    for stage in stages:
        if stage not in ALL_STAGES:
            raise SystemExit(f"unknown stage: {stage}")

    CLOUD_ROOT.mkdir(parents=True, exist_ok=True)
    problems = []
    for name in selected:
        if "prepare" in stages:
            backbone_configs(name, specs[name])
        problems.extend(check_backbone(name, specs[name]))

    commands = build_commands(selected, specs, stages, args.limit)
    command_file = CLOUD_ROOT / "commands.sh"
    command_file.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(command(argv) for _, argv in commands) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "finplan-cloud-suite-v1",
        "repo": str(REPO_ROOT),
        "backbones": selected,
        "stages": list(stages),
        "limit": args.limit,
        "python": sys.executable,
        "commands_file": str(command_file),
        "commands": [{"stage": stage, "command": command(argv)} for stage, argv in commands],
        "problems": problems,
        "execute": bool(args.execute),
    }
    (CLOUD_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2)[:6000])

    if not args.execute or args.dry_run:
        print("\nDry run. Review paper/results/cloud/commands.sh, then pass --execute.")
        return
    if problems:
        print("\nEnvironment problems must be fixed before execution:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        raise SystemExit(2)

    for stage, argv in commands:
        print(f"\n[{stage}] {command(argv)}", flush=True)
        key = stage.split(":")[0]
        env = command_env(key, specs[key]) if key in specs else dict(os.environ)
        subprocess.run(argv, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
