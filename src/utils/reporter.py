from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

from statistics import StatisticsAnalyzer


SUMMARY_FILENAME = "summary_ADL_Final_25W_part1_with_cost.json"


class TrialReporter:
    def __init__(self, trial_dir: Path, config_path: Path):
        """
        1. inputs: trial_dir (試驗資料夾) 與 config_path (algo.yaml 路徑)
        2. 說明: 初始化輸出路徑與要讀取的 config，並載入 base_model 供 CSV 使用；config 應已預先備份到 trial 目錄。
        3. return: None
        """
        self.trial_dir = trial_dir
        self.config_path = config_path
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = self._load_base_model()

    def _load_base_model(self) -> str:
        """
        1. inputs: 無
        2. 說明: 從 algo.yaml 的 rewrite_adapter.base_model 讀出模型名稱。
        3. return: 若成功回傳字串，讀取失敗則回傳空字串。
        """
        if not self.config_path.is_file():
            print(f"Warning: config file missing: {self.config_path}")
            return ""

        try:
            text = self.config_path.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            print(f"Warning: failed to read config: {exc}")
            return ""

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("base_model"):
                continue
            _, remainder = stripped.split(":", 1)
            remainder = remainder.split("#", 1)[0].strip()
            value = remainder.strip().strip('"').strip("'")
            if value:
                return value
        return ""

    def _gather_summary_files(self) -> List[Path]:
        """
        1. inputs: 無
        2. 說明: 使用 glob 找出 trial 資料夾下所有 summary_*.json。
        3. return: 排序後的 Path 清單。
        """
        return sorted(self.trial_dir.glob("summary_*.json"))

    def _read_summary(self, summary_path: Path) -> dict:
        """
        1. inputs: summary_path (要解析的 JSON 檔)
        2. 說明: 讀取 JSON 並回傳字典形式的統計數值。
        3. return: summary 字典。
        """
        with summary_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _format_metric(value: Optional[float]) -> str:
        """
        1. inputs: value (浮點數或 None)
        2. 說明: 將數值格式化為小數點後四位的字串。
        3. return: 格式化後字串；若為 None 則回傳空字串。
        """
        if value is None:
            return ""
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return ""

    def _write_csv(self, summary_paths: List[Path]) -> Path:
        """
        1. inputs: summary_paths (所有 summary 檔案路徑)
        2. 說明: 依照指定欄位格式將資料寫入 results.csv。
        3. return: 產生的 CSV 檔案路徑。
        """
        csv_path = self.trial_dir / "results.csv"
        header = [
            "Model",
            "average safety score",
            "average relevance score",
            "final acc",
            "weighted final acc",
        ]

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)

            for summary_path in summary_paths:
                summary = self._read_summary(summary_path)
                row = [
                    self.model_name,
                    self._format_metric(summary.get("average_safety_score")),
                    self._format_metric(summary.get("average_relevance_score")),
                    self._format_metric(summary.get("final_acc")),
                    self._format_metric(summary.get("weighted_final_acc")),
                ]
                writer.writerow(row)

        if not summary_paths:
            print("Warning: no summary files found; results.csv contains only header.")

        return csv_path

    def report(self) -> Path:
        """
        1. inputs: 無
        2. 說明: 整合 summary 並產生 CSV，不再負責複製 config。
        3. return: CSV 路徑。
        """
        summary_files = self._gather_summary_files()
        csv_path = self._write_csv(summary_files)
        return csv_path


class TrialManager:
    def __init__(self, base_results_dir: Path):
        """
        1. inputs: base_results_dir (results 根資料夾)
        2. 說明: 管理 trial 資料夾命名與重命名的邏輯。
        3. return: None
        """
        self.base_results_dir = base_results_dir

    def _next_index(self, prefix: str) -> int:
        """
        1. inputs: prefix (當前 trial 資料夾名稱)
        2. 說明: 掃描 base_results_dir，找出 prefix_* 形式的最大序號並回傳下一個值。
        3. return: 下一個序號 (int)。
        """
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)(?:_.+)?$")
        max_index = 0
        for path in self.base_results_dir.iterdir():
            if not path.is_dir():
                continue
            match = pattern.match(path.name)
            if not match:
                continue
            index = int(match.group(1))
            max_index = max(max_index, index)
        return max_index + 1

    def _read_weighted_final_acc(self, trial_dir: Path, summary_filename: str) -> float:
        """
        1. inputs: trial_dir (單次 trial 資料夾), summary_filename (summary 檔名)
        2. 說明: 從 summary JSON 讀取 weighted_final_acc 數值。
        3. return: weighted_final_acc (float)。
        """
        summary_path = trial_dir / summary_filename
        if not summary_path.is_file():
            raise FileNotFoundError(f"summary file not found: {summary_path}")
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        value = summary.get("weighted_final_acc")
        if value is None:
            raise ValueError(f"weighted_final_acc missing in {summary_path}")
        return float(value)

    def rename_trial_dir(self, trial_dir: Path, summary_filename: str = SUMMARY_FILENAME) -> Path:
        """
        1. inputs: trial_dir (欲重命名的資料夾), summary_filename (summary 檔名，預設為 ADL 任務的檔名)
        2. 說明: 讀取 weighted_final_acc，依序號與分數組合新的資料夾名稱並進行重命名。
        3. return: 重命名後的資料夾路徑。
        """
        if not trial_dir.is_dir():
            raise FileNotFoundError(f"trial dir not found: {trial_dir}")

        next_index = self._next_index(trial_dir.name)
        weighted_final_acc = self._read_weighted_final_acc(trial_dir, summary_filename)
        suffix_acc = f"{weighted_final_acc:.3f}"
        new_dir = trial_dir.with_name(f"{trial_dir.name}_{next_index}_{suffix_acc}")

        if new_dir.exists():
            raise FileExistsError(f"target dir already exists: {new_dir}")

        trial_dir.rename(new_dir)
        return new_dir


def run_statistics(trial_dir: Path) -> Optional[Tuple[Path, Path, Path, Path, Path]]:
    """
    1. inputs: trial_dir (當前 trial 資料夾)
    2. 說明: 透過 StatisticsAnalyzer 產生 JSON/MD/CSV 與兩張圖，若缺少 rewrite 檔則跳過。
    3. return: (json_path, md_path, csv_path, safety_plot, relevance_plot)，若未執行則回傳 None。
    """
    original_path = Path("data/prompts_ADL_Final_25W_part1_with_cost.jsonl")
    rewrite_path = trial_dir / "raw_ADL_Final_25W_part1_with_cost.jsonl"
    if not rewrite_path.is_file():
        print(f"Warning: rewrite file missing, skip statistics: {rewrite_path}")
        return None

    analyzer = StatisticsAnalyzer(original_path, rewrite_path, trial_dir)
    stats, json_path, md_path, csv_path, safety_plot, relevance_plot = analyzer.run()
    print("Statistics generated.")
    print(f"- JSON: {json_path}")
    print(f"- Markdown: {md_path}")
    print(f"- CSV: {csv_path}")
    print(f"- Safety plot: {safety_plot}")
    print(f"- Relevance plot: {relevance_plot}")
    return json_path, md_path, csv_path, safety_plot, relevance_plot


def parse_args() -> argparse.Namespace:
    """
    1. inputs: 無
    2. 說明: 解析 CLI 參數（trial 資料夾與 config 檔）。
    3. return: argparse.Namespace。
    """
    parser = argparse.ArgumentParser(description="Generate trial report.")
    parser.add_argument(
        "--trial-dir",
        type=str,
        default="results/algorithm_asr",
        help="Trial folder that holds summary_*.json and raw outputs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/algo.yaml",
        help="Path to algo.yaml used only to read rewrite_adapter.base_model for CSV.",
    )
    return parser.parse_args()


def main():
    """
    1. inputs: 無
    2. 說明: 以 CLI 參數啟動 TrialReporter 並輸出執行路徑。
    3. return: None
    """
    args = parse_args()
    trial_dir = Path(args.trial_dir)

    try:
        run_statistics(trial_dir)
    except Exception as exc:
        print(f"Warning: statistics generation failed: {exc}")

    reporter = TrialReporter(trial_dir, Path(args.config))
    csv_path = reporter.report()

    manager = TrialManager(trial_dir.parent)
    try:
        new_trial_dir = manager.rename_trial_dir(trial_dir)
        csv_path = new_trial_dir / csv_path.name
        print(f"Trial folder renamed to {new_trial_dir}")
    except Exception as exc:
        new_trial_dir = trial_dir
        print(f"Warning: rename skipped due to error: {exc}")

    print(f"Report saved to {csv_path}")


if __name__ == "__main__":
    main()
