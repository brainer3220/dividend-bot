"""
경제 유동성 지표 분석 전략 — 전체 파일 버전 v2 (그리드 탐색 파라미터 포함)

v2 핵심 추가/변경
1) ✅ Risk-off에서 SPY 잔존 익스포저를 그리드로 탐색 (risk_off_spy_grid)
   - Risk-off에서 SPY를 줄이면 MDD가 직접 감소(대신 CAGR/Sharpe 일부 희생 가능)
2) ✅ Trend filter 강화 옵션(그리드): price>MA AND (MA_slope>0) 사용 여부 (trend_slope_grid)
3) ✅ Crash lockout(그리드): 급락(-7% 등) 발생 시 N주 강제 Risk-off (crash_enable_grid 등)
4) ✅ CV 선택 점수 개선: 평균이 아니라 "worst-fold"를 점수에 반영
   - Score = w_sharpe * WorstSharpe + w_calmar * MinCalmar - w_switch * MeanSwitchesPerYear - w_worst_dd * abs(WorstMaxDD)
5) ✅ 캐시 안정화: 지표/티커 조합별 hash key 캐시 + 컬럼 검증 후 자동 재다운로드
6) ✅ trial마다 전체기간 1회 backtest(out_full) → fold별 val 슬라이스 평가 (히스토리 단절 방지)
7) ✅ 로그/체크포인트/스모크테스트/드로우다운 리포트 포함

설치:
pip install pandas numpy yfinance pandas_datareader tqdm pyarrow

실행:
python liquidity_strategy_full_v2.py

출력:
./checkpoints/
  - cv_results.csv
  - final_backtest.parquet
  - final_stats.json
  - final_drawdown_report.json
./logs/run.log
"""

from __future__ import annotations

import os
import json
import math
import pickle
import logging
import warnings
import hashlib
import time
import argparse
import traceback
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# =============================
# Logging
# =============================
def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("liquidity_strategy_v2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(log_dir, "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# =============================
# Config
# =============================
@dataclass(frozen=True)
class DataConfig:
    start: str = "2005-01-01"
    end: str = "2025-12-31"

    cache_dir: str = "./cache"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    fred_indicators: Tuple[str, ...] = (
        "WALCL",
        "WRESBAL",
        "RRPONTSYD",
        "M2SL",
        # "WTREGEN",  # 필요 시 추가
    )

    assets: Tuple[str, ...] = (
        "SPY",
        "BIL",
        "GLD",
        "BTC-USD",
    )


@dataclass(frozen=True)
class FeatureConfig:
    freq: str = "W-FRI"
    transform: str = "diff"  # "diff" or "pct"
    zwin: int = 52
    smooth_span: int = 8
    clip_z: float = 6.0
    ffill_limit: int = 2
    use_multi_timeframe: bool = False
    multi_zwin: Tuple[int, ...] = (26, 52, 104)
    multi_smooth: Tuple[int, ...] = (4, 8, 12)
    multi_agg: str = "mean"


@dataclass(frozen=True)
class BaseSignalConfig:
    # trial에서 threshold/hysteresis/cooldown/trend/crash를 덮어씀
    threshold: float = 0.0
    hysteresis: float = 0.2
    cooldown_weeks: int = 8
    adaptive: bool = False
    adaptive_lookback: int = 52
    adaptive_k: float = 0.5


@dataclass(frozen=True)
class BaseTrendConfig:
    trend_ticker: str = "SPY"
    trend_win: int = 40              # 기본값(그리드에서 덮어씀)
    use_slope_filter: bool = False   # 기본값(그리드에서 덮어씀)


@dataclass(frozen=True)
class BaseCrashConfig:
    enabled: bool = False
    crash_ret: float = -0.07
    lock_weeks: int = 8
    consecutive_threshold: int = 3


@dataclass(frozen=True)
class PortfolioConfig:
    # DataConfig.assets 순서와 동일해야 함 (SPY, BIL, GLD, BTC-USD)
    w_risk_on: Tuple[float, ...] = (0.80, 0.10, 0.10, 0.00)

    # Risk-off는 trial에서 SPY 비중을 바꾸고, BIL을 자동 조정(합=1 유지)
    risk_off_spy: float = 0.20
    risk_off_gld: float = 0.15
    risk_off_btc: float = 0.00

    fee_bps: float = 2.0
    slippage_bps: float = 5.0
    fee_by_asset: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        assert abs(sum(self.w_risk_on) - 1.0) < 1e-6, "Risk-on weights must sum to 1"
        assert all(w >= 0 for w in self.w_risk_on), "Weights must be non-negative"
        assert 0.0 <= self.risk_off_spy <= 1.0, "SPY weight must be in [0,1]"
        assert 0.0 <= self.risk_off_gld <= 1.0, "GLD weight must be in [0,1]"
        assert 0.0 <= self.risk_off_btc <= 1.0, "BTC weight must be in [0,1]"
        assert (self.risk_off_spy + self.risk_off_gld + self.risk_off_btc) <= 1.0, (
            "Risk-off weights sum exceeds 1"
        )


@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 5
    min_train_weeks: int = 156
    min_val_weeks: int = 26

    # ✅ worst-fold 기반 선택
    w_sharpe: float = 1.0
    w_calmar: float = 0.5
    w_switch: float = 0.10
    w_worst_dd: float = 0.25

    sort_by: str = "Score"  # "Score" 추천
    batch_size: int = 100
    max_workers: int = 1
    use_process_pool: bool = False
    use_staged_search: bool = False


@dataclass(frozen=True)
class SearchSpaceConfig:
    # === LCI/레짐 파라미터(기본 27) ===
    zwin_grid: Tuple[int, ...] = (26, 52, 78)
    smooth_grid: Tuple[int, ...] = (4, 8, 12)
    threshold_grid: Tuple[float, ...] = (-0.25, 0.0, 0.25)

    # === 레짐 안정화 ===
    hysteresis_grid: Tuple[float, ...] = (0.1, 0.2)      # 필요 시 (0.0,0.1,0.2,0.3)
    cooldown_grid: Tuple[int, ...] = (0, 8)              # 필요 시 (0,4,8,12)

    # === 추세 필터 ===
    trend_win_grid: Tuple[int, ...] = (20, 40)           # 0 포함하면 비활성화 가능
    trend_slope_grid: Tuple[int, ...] = (0, 1)           # 0: slope 미사용, 1: slope 사용

    # === Crash lockout ===
    crash_enable_grid: Tuple[int, ...] = (0, 1)
    crash_ret_grid: Tuple[float, ...] = (-0.07,)         # 필요 시 (-0.05,-0.07,-0.10)
    crash_lock_grid: Tuple[int, ...] = (8,)              # 필요 시 (4,8,12)

    # === Risk-off SPY 비중 탐색 ===
    risk_off_spy_grid: Tuple[float, ...] = (0.00, 0.05, 0.10)  # MDD 줄이려면 낮춰야 함
    # GLD는 고정(PortfolioConfig.risk_off_gld)로 두는 것을 권장


# =============================
# Utilities
# =============================
def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _assert_sorted_index(x: pd.Series | pd.DataFrame) -> None:
    assert isinstance(x.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
    assert x.index.is_monotonic_increasing, "Index must be sorted ascending"
    assert x.index.has_duplicates is False, "Index must not contain duplicates"


def _to_parquet(df: pd.DataFrame, path: str) -> None:
    _safe_mkdir(os.path.dirname(path))
    tmp = path + ".tmp"
    df.to_parquet(tmp)
    os.replace(tmp, path)


def _read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _save_pickle(obj: object, path: str) -> None:
    _safe_mkdir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def cache_key(items: List[str]) -> str:
    s = ",".join(sorted(items))
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def switches_count(regime: pd.Series) -> int:
    _assert_sorted_index(regime.to_frame("regime"))
    return int((regime.diff().fillna(0).abs() > 0).sum())


def switches_per_year(regime: pd.Series, periods_per_year: int = 52) -> float:
    n = float(len(regime))
    assert n > 0
    years = n / float(periods_per_year)
    return float(switches_count(regime) / max(years, 1e-12))


# =============================
# CLI & Config helpers
# =============================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Liquidity Strategy Backtesting System")

    parser.add_argument(
        "--mode",
        type=str,
        default="backtest",
        choices=["backtest", "live", "optimize", "monte_carlo", "profile"],
        help="Operation mode",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--start", type=str, default="2005-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--walk-forward", action="store_true", help="Use walk-forward optimization")
    parser.add_argument("--staged", action="store_true", help="Use staged grid search")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--notify", action="store_true", help="Enable Telegram notifications")
    parser.add_argument("--mc-sims", type=int, default=200, help="Number of Monte Carlo simulations")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory")

    return parser.parse_args()


def _as_tuple(values: object, cast_type: Optional[type] = None) -> Tuple[object, ...]:
    if values is None:
        return tuple()
    if isinstance(values, tuple):
        return values
    if isinstance(values, list):
        if cast_type is None:
            return tuple(values)
        return tuple(cast_type(v) for v in values)
    raise TypeError(f"Expected list/tuple, got {type(values)}")


def _merge_dataclass(base: object, updates: Dict[str, object], tuple_fields: Optional[Dict[str, type]] = None) -> object:
    tuple_fields = tuple_fields or {}
    base_dict = asdict(base)
    for k, v in updates.items():
        if k in tuple_fields and v is not None:
            base_dict[k] = _as_tuple(v, tuple_fields[k])
        else:
            base_dict[k] = v
    for k, t in tuple_fields.items():
        if k in base_dict:
            base_dict[k] = _as_tuple(base_dict[k], t)
    return type(base)(**base_dict)


def build_configs(args: argparse.Namespace, config_dict: Optional[Dict[str, object]]) -> Tuple[
    DataConfig, FeatureConfig, BaseSignalConfig, BaseTrendConfig, BaseCrashConfig, PortfolioConfig, CVConfig, SearchSpaceConfig
]:
    dcfg = DataConfig()
    fcfg = FeatureConfig()
    base_scfg = BaseSignalConfig()
    base_tcfg = BaseTrendConfig()
    base_ccfg = BaseCrashConfig()
    base_pcfg = PortfolioConfig()
    cvcfg = CVConfig()
    spcfg = SearchSpaceConfig()

    if config_dict:
        dcfg = _merge_dataclass(
            dcfg,
            config_dict.get("data", {}),
            {"fred_indicators": str, "assets": str},
        )
        fcfg = _merge_dataclass(fcfg, config_dict.get("feature", {}))
        base_scfg = _merge_dataclass(base_scfg, config_dict.get("signal", {}))
        base_tcfg = _merge_dataclass(base_tcfg, config_dict.get("trend", {}))
        base_ccfg = _merge_dataclass(base_ccfg, config_dict.get("crash", {}))
        base_pcfg = _merge_dataclass(base_pcfg, config_dict.get("portfolio", {}))
        cvcfg = _merge_dataclass(cvcfg, config_dict.get("cv", {}))
        spcfg = _merge_dataclass(
            spcfg,
            config_dict.get("search", {}),
            {
                "zwin_grid": int,
                "smooth_grid": int,
                "threshold_grid": float,
                "hysteresis_grid": float,
                "cooldown_grid": int,
                "trend_win_grid": int,
                "trend_slope_grid": int,
                "crash_enable_grid": int,
                "crash_ret_grid": float,
                "crash_lock_grid": int,
                "risk_off_spy_grid": float,
            },
        )

    dcfg = replace(dcfg, start=args.start, end=args.end, checkpoint_dir=args.output_dir)
    cvcfg = replace(cvcfg, max_workers=int(args.workers), use_staged_search=bool(args.staged))
    return dcfg, fcfg, base_scfg, base_tcfg, base_ccfg, base_pcfg, cvcfg, spcfg


def load_best_params_from_cv(
    dcfg: DataConfig,
    fallback_params: Optional[TrialParams] = None,
) -> TrialParams:
    def _trial_params_from_dict(best: Dict[str, object], prefix: str = "param_") -> TrialParams:
        key = lambda name: f"{prefix}{name}" if prefix else name
        try:
            return TrialParams(
                zwin=int(best[key("zwin")]),
                smooth_span=int(best[key("smooth_span")]),
                threshold=float(best[key("threshold")]),
                hysteresis=float(best[key("hysteresis")]),
                cooldown_weeks=int(best[key("cooldown_weeks")]),
                trend_win=int(best[key("trend_win")]),
                use_trend_slope=int(best[key("use_trend_slope")]),
                crash_enabled=int(best[key("crash_enabled")]),
                crash_ret=float(best[key("crash_ret")]),
                crash_lock_weeks=int(best[key("crash_lock_weeks")]),
                risk_off_spy=float(best[key("risk_off_spy")]),
            )
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Best params missing required key: {exc.args[0]}") from exc

    _safe_mkdir(dcfg.checkpoint_dir)
    cv_path = os.path.join(dcfg.checkpoint_dir, "cv_results.csv")
    if not os.path.exists(cv_path):
        stats_path = os.path.join(dcfg.checkpoint_dir, "final_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            best_params = stats.get("best_params")
            if best_params:
                return _trial_params_from_dict(best_params, prefix="")
        if fallback_params is not None:
            logging.getLogger("liquidity_strategy_v2").warning(
                "cv_results.csv not found at %s; using fallback params from config.", cv_path
            )
            return fallback_params
        raise RuntimeError(
            "cv_results.csv not found and no cached best_params available. "
            "Run in optimize mode first to generate checkpoints."
        )
    df = pd.read_csv(cv_path)
    if df.empty:
        raise RuntimeError("cv_results.csv is empty")
    best = df.iloc[0].to_dict()
    return _trial_params_from_dict(best, prefix="param_")


def default_params_from_configs(
    fcfg: FeatureConfig,
    base_scfg: BaseSignalConfig,
    base_tcfg: BaseTrendConfig,
    base_ccfg: BaseCrashConfig,
    base_pcfg: PortfolioConfig,
) -> TrialParams:
    return TrialParams(
        zwin=int(fcfg.zwin),
        smooth_span=int(fcfg.smooth_span),
        threshold=float(base_scfg.threshold),
        hysteresis=float(base_scfg.hysteresis),
        cooldown_weeks=int(base_scfg.cooldown_weeks),
        trend_win=int(base_tcfg.trend_win),
        use_trend_slope=1 if base_tcfg.use_slope_filter else 0,
        crash_enabled=1 if base_ccfg.enabled else 0,
        crash_ret=float(base_ccfg.crash_ret),
        crash_lock_weeks=int(base_ccfg.lock_weeks),
        risk_off_spy=float(base_pcfg.risk_off_spy),
    )


def get_notification_config(
    config_dict: Optional[Dict[str, object]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    cfg = {}
    if config_dict:
        cfg = config_dict.get("notifications", {})
    enabled = bool(cfg.get("enabled", False)) or bool(args.notify)
    cfg = dict(cfg)
    cfg["enabled"] = enabled
    cfg.setdefault("rules", {})
    return cfg

# =============================
# Data Loaders
# =============================
def fetch_fred(
    indicators: List[str],
    start: str,
    end: str,
    cache_path: str,
    logger: logging.Logger
) -> pd.DataFrame:
    _safe_mkdir(os.path.dirname(cache_path))

    if os.path.exists(cache_path):
        df = _read_parquet(cache_path)
        _assert_sorted_index(df)
        if set(indicators).issubset(set(df.columns)):
            logger.info(f"Load FRED cache: {cache_path}")
            out = df.loc[start:end, indicators].copy()
            _assert_sorted_index(out)
            return out
        logger.info(f"FRED cache mismatch -> redownload. cache={list(df.columns)} req={indicators}")

    logger.info("Download FRED data (no valid cache found).")
    try:
        from pandas_datareader import data as pdr
    except Exception as e:
        raise RuntimeError("pandas_datareader 필요: pip install pandas_datareader") from e

    frames = []
    for code in tqdm(indicators, desc="FRED download"):
        s = pdr.DataReader(code, "fred", start, end)  # type: ignore
        assert isinstance(s, pd.DataFrame) and s.shape[1] == 1
        s.columns = [code]
        frames.append(s)

    df = pd.concat(frames, axis=1).sort_index()
    _assert_sorted_index(df)
    assert set(indicators).issubset(set(df.columns)), "FRED download missing indicators"

    _to_parquet(df, cache_path)
    logger.info(f"Saved FRED cache: {cache_path}")

    out = df.loc[start:end, indicators].copy()
    _assert_sorted_index(out)
    return out


def fetch_prices_yf(
    tickers: List[str],
    start: str,
    end: str,
    cache_path: str,
    logger: logging.Logger
) -> pd.DataFrame:
    _safe_mkdir(os.path.dirname(cache_path))

    if os.path.exists(cache_path):
        df = _read_parquet(cache_path)
        _assert_sorted_index(df)
        if set(tickers).issubset(set(df.columns)):
            logger.info(f"Load price cache: {cache_path}")
            out = df.loc[start:end, tickers].copy()
            _assert_sorted_index(out)
            return out
        logger.info(f"Price cache mismatch -> redownload. cache={list(df.columns)} req={tickers}")

    logger.info("Download price data via yfinance (no valid cache found).")
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance 필요: pip install yfinance") from e

    max_retries = 3
    data = None
    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
            if data is None or getattr(data, "empty", False):
                raise RuntimeError("yfinance returned empty data")
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Download failed (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)

    if isinstance(data.columns, pd.MultiIndex):
        closes = []
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                continue
            c = data[(t, "Close")].rename(t)
            closes.append(c)
        df = pd.concat(closes, axis=1) if closes else pd.DataFrame(index=data.index)
    else:
        assert len(tickers) == 1, "Unexpected yfinance output"
        df = data["Close"].to_frame(tickers[0])

    df = df.sort_index()
    _assert_sorted_index(df)

    missing = [t for t in tickers if t not in df.columns]
    if missing:
        raise RuntimeError(
            f"yfinance 결과에 없는 티커: {missing}. "
            f"티커 오타/일시 누락/응답 이슈 가능. tickers={tickers}"
        )

    all_na = [t for t in tickers if df[t].notna().sum() == 0]
    if all_na:
        raise RuntimeError(f"전 구간 NaN 티커: {all_na}. 기간/티커를 조정하세요.")

    _to_parquet(df, cache_path)
    logger.info(f"Saved price cache: {cache_path}")

    out = df.loc[start:end, tickers].copy()
    _assert_sorted_index(out)
    return out


# =============================
# Preprocess
# =============================
def resample_to_freq(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    _assert_sorted_index(df)
    out = df.resample(freq).last()
    _assert_sorted_index(out)
    return out


def align_data(
    fred_w: pd.DataFrame,
    prices_w: pd.DataFrame,
    ffill_limit_fred: int,
    min_non_na_assets: int = 2,
    min_len: int = 200,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _assert_sorted_index(fred_w)
    _assert_sorted_index(prices_w)
    assert ffill_limit_fred >= 0

    fred_w2 = fred_w.ffill(limit=ffill_limit_fred)
    _assert_sorted_index(fred_w2)

    valid_rows = (prices_w.notna().sum(axis=1) >= min_non_na_assets)
    prices_w2 = prices_w.loc[valid_rows].copy()
    _assert_sorted_index(prices_w2)

    idx = fred_w2.index.intersection(prices_w2.index).sort_values()
    assert len(idx) >= min_len, f"Aligned dataset too short: {len(idx)} < {min_len}"

    fred_a = fred_w2.loc[idx].copy()
    prices_a = prices_w2.loc[idx].copy()

    fred_na_ratio = float(fred_a.isna().sum().sum()) / max(float(fred_a.size), 1.0)
    if fred_na_ratio > 0.10 and logger is not None:
        logger.warning(f"FRED data has >10% NaN after ffill: {fred_na_ratio:.1%}")

    if prices_a.isna().any().any() and logger is not None:
        na_counts = prices_a.isna().sum()
        na_map = {k: int(v) for k, v in na_counts[na_counts > 0].to_dict().items()}
        logger.warning(f"Price data still has NaN: {na_map}")

    assert prices_a.notna().sum().min() > 0, "Some asset has no data at all"
    return fred_a, prices_a


def load_and_align_data(
    dcfg: DataConfig,
    fcfg: FeatureConfig,
    logger: logging.Logger,
    cache_prefix: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fred_key = cache_key(list(dcfg.fred_indicators))
    px_key = cache_key(list(dcfg.assets))

    fred = fetch_fred(
        indicators=list(dcfg.fred_indicators),
        start=dcfg.start,
        end=dcfg.end,
        cache_path=os.path.join(dcfg.cache_dir, f"{cache_prefix}fred_{fred_key}.parquet"),
        logger=logger,
    )
    prices = fetch_prices_yf(
        tickers=list(dcfg.assets),
        start=dcfg.start,
        end=dcfg.end,
        cache_path=os.path.join(dcfg.cache_dir, f"{cache_prefix}prices_{px_key}.parquet"),
        logger=logger,
    )

    fred_w = resample_to_freq(fred, fcfg.freq)
    prices_w = resample_to_freq(prices, fcfg.freq)

    fred_a, prices_a = align_data(
        fred_w=fred_w,
        prices_w=prices_w,
        ffill_limit_fred=fcfg.ffill_limit,
        min_non_na_assets=2,
        min_len=200,
        logger=logger,
    )
    assert list(prices_a.columns) == list(dcfg.assets)
    return fred_a, prices_a


# =============================
# Feature Engineering (LCI)
# =============================
def transform_indicators(df: pd.DataFrame, transform: str) -> pd.DataFrame:
    assert transform in ("diff", "pct"), "transform must be diff or pct"
    _assert_sorted_index(df)
    out = df.diff() if transform == "diff" else df.pct_change()
    _assert_sorted_index(out)
    return out


def rolling_zscore(df: pd.DataFrame, win: int, clip_z: float) -> pd.DataFrame:
    assert win >= 10, "zwin too small"
    _assert_sorted_index(df)

    mu = df.rolling(win).mean()
    sd = df.rolling(win).std(ddof=0)
    z = (df - mu) / sd.replace(0, np.nan)

    z = z.clip(lower=-clip_z, upper=clip_z)
    _assert_sorted_index(z)
    return z


def ema_smooth(s: pd.Series, span: int) -> pd.Series:
    assert span >= 1
    out = s.ewm(span=span, adjust=False).mean()
    _assert_sorted_index(out.to_frame("x"))
    return out


def build_lci(
    fred_w: pd.DataFrame,
    fcfg: FeatureConfig,
    weights: Optional[Dict[str, float]] = None
) -> pd.Series:
    _assert_sorted_index(fred_w)
    assert fred_w.shape[1] >= 2

    x_t = transform_indicators(fred_w, fcfg.transform)
    z = rolling_zscore(x_t, fcfg.zwin, fcfg.clip_z)

    if weights is None:
        weights = {
            "WALCL": +1.0,
            "WRESBAL": +0.8,
            "RRPONTSYD": -1.0,
            "M2SL": +0.6,
        }

    cols = [c for c in z.columns if c in weights]
    assert len(cols) >= 2, "Not enough weighted indicators found"

    w = np.array([weights[c] for c in cols], dtype=float)
    assert np.isfinite(w).all()

    denom = float(np.sum(np.abs(w)) + 1e-12)
    assert denom > 0.0
    w = w / denom

    lci_raw = (z[cols].fillna(0.0) * w).sum(axis=1)
    lci = ema_smooth(lci_raw, fcfg.smooth_span).rename("LCI")
    lci_std = float(lci.std(ddof=0))
    if lci_std < 0.1:
        warnings.warn(
            f"LCI has very low volatility ({lci_std:.3f}); regime switching may be rare",
            RuntimeWarning,
        )
    _assert_sorted_index(lci.to_frame("LCI"))
    return lci


def build_multi_timeframe_lci(
    fred_w: pd.DataFrame,
    fcfg: FeatureConfig,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    _assert_sorted_index(fred_w)
    assert len(fcfg.multi_zwin) == len(fcfg.multi_smooth), "multi_zwin/multi_smooth length mismatch"
    assert len(fcfg.multi_zwin) >= 2, "multi-timeframe requires >=2 windows"

    lcis: List[pd.Series] = []
    for zwin, smooth in zip(fcfg.multi_zwin, fcfg.multi_smooth):
        fcfg_i = FeatureConfig(
            freq=fcfg.freq,
            transform=fcfg.transform,
            zwin=int(zwin),
            smooth_span=int(smooth),
            clip_z=fcfg.clip_z,
            ffill_limit=fcfg.ffill_limit,
            use_multi_timeframe=False,
            multi_zwin=fcfg.multi_zwin,
            multi_smooth=fcfg.multi_smooth,
            multi_agg=fcfg.multi_agg,
        )
        lci_i = build_lci(fred_w, fcfg_i, weights=weights)
        lci_i = lci_i.rename(f"LCI_{int(zwin)}_{int(smooth)}")
        lcis.append(lci_i)

    out = pd.concat(lcis, axis=1)
    _assert_sorted_index(out)
    return out


# =============================
# Signal (Regime)
# =============================
@dataclass(frozen=True)
class SignalConfig:
    threshold: float
    hysteresis: float
    cooldown_weeks: int
    adaptive: bool
    adaptive_lookback: int
    adaptive_k: float


def regime_from_lci(lci: pd.Series, scfg: SignalConfig) -> pd.Series:
    """
    LCI로부터 risk-on/off 레짐을 생성.

    Parameters
    ----------
    lci : pd.Series
        Liquidity Condition Index
    scfg : SignalConfig
        threshold: 레짐 전환 기준값
        hysteresis: 히스테리시스 폭
        cooldown_weeks: 최소 유지 기간
        adaptive: 동적 threshold 사용 여부
        adaptive_lookback: rolling window
        adaptive_k: mean ± k*std
    """
    _assert_sorted_index(lci.to_frame("LCI"))

    thr = float(scfg.threshold)
    h = float(scfg.hysteresis)
    cd = int(scfg.cooldown_weeks)
    adaptive = bool(scfg.adaptive)
    lookback = int(scfg.adaptive_lookback)
    k = float(scfg.adaptive_k)

    assert h >= 0.0
    assert cd >= 0
    if adaptive:
        assert lookback >= 10
        assert k >= 0.0

    regime = pd.Series(index=lci.index, dtype=int, name="regime")

    state = 0
    since_switch = 10**9

    upper = None
    lower = None
    if adaptive:
        mu = lci.rolling(lookback).mean()
        sd = lci.rolling(lookback).std(ddof=0)
        upper = mu + k * sd
        lower = mu - k * sd

    for i, v in enumerate(lci.values):
        since_switch += 1

        if not np.isfinite(v):
            regime.iloc[i] = state
            continue

        if since_switch < cd:
            regime.iloc[i] = state
            continue

        new_state = state
        if adaptive:
            up_i = float(upper.iloc[i]) if upper is not None else np.nan
            lo_i = float(lower.iloc[i]) if lower is not None else np.nan
            if not np.isfinite(up_i) or not np.isfinite(lo_i):
                regime.iloc[i] = state
                continue
            if h == 0.0:
                if v > up_i:
                    new_state = 1
                elif v < lo_i:
                    new_state = 0
            else:
                if state == 0 and v > up_i + h:
                    new_state = 1
                elif state == 1 and v < lo_i - h:
                    new_state = 0
        else:
            if h == 0.0:
                new_state = 1 if v > thr else 0
            else:
                if state == 0 and v > thr + h:
                    new_state = 1
                elif state == 1 and v < thr - h:
                    new_state = 0

        if new_state != state:
            state = new_state
            since_switch = 0

        regime.iloc[i] = state

    assert set(regime.unique()).issubset({0, 1})
    _assert_sorted_index(regime.to_frame("regime"))
    return regime


def apply_trend_filter(
    regime: pd.Series,
    prices_w: pd.DataFrame,
    tcfg: BaseTrendConfig,
    trend_win: int,
    use_slope_filter: bool,
) -> pd.Series:
    _assert_sorted_index(regime.to_frame("regime"))
    _assert_sorted_index(prices_w)

    win = int(trend_win)
    if win <= 0:
        return regime.astype(int)

    tkr = str(tcfg.trend_ticker)
    assert tkr in prices_w.columns, f"Trend ticker missing in prices: {tkr}"

    p = prices_w[tkr].reindex(regime.index)
    ma = p.rolling(win).mean()

    if use_slope_filter:
        trend_ok = (p > ma) & (ma.diff() > 0)
    else:
        trend_ok = (p > ma)

    trend_ok = trend_ok.fillna(False)

    regime2 = (regime.astype(int) * trend_ok.astype(int)).astype(int)
    regime2.name = "regime"

    _assert_sorted_index(regime2.to_frame("regime"))
    return regime2


def apply_crash_lockout(
    regime: pd.Series,
    prices_w: pd.DataFrame,
    enabled: bool,
    crash_ret: float,
    lock_weeks: int,
    tkr: str = "SPY",
    consecutive_threshold: int = 3,
) -> pd.Series:
    _assert_sorted_index(regime.to_frame("regime"))
    _assert_sorted_index(prices_w)

    if not enabled:
        return regime.astype(int)

    assert tkr in prices_w.columns, f"Crash ticker missing in prices: {tkr}"
    assert lock_weeks >= 1
    assert crash_ret < 0
    assert consecutive_threshold >= 1

    p = prices_w[tkr].reindex(regime.index)
    r = p.pct_change().fillna(0.0)

    out = regime.astype(int).copy()
    lock = 0
    consecutive_drops = 0

    for i, dt in enumerate(out.index):
        if lock > 0:
            out.iloc[i] = 0
            lock -= 1
            continue

        ret_i = float(r.loc[dt])
        if ret_i < 0:
            consecutive_drops += 1
        else:
            consecutive_drops = 0

        crash_by_drop = ret_i <= float(crash_ret)
        crash_by_streak = consecutive_threshold > 1 and consecutive_drops >= consecutive_threshold
        if crash_by_drop or crash_by_streak:
            out.iloc[i] = 0
            lock = int(lock_weeks)
            consecutive_drops = 0

    out.name = "regime"
    _assert_sorted_index(out.to_frame("regime"))
    return out


# =============================
# Portfolio weights helper
# =============================
def build_portfolio_weights(
    base_pcfg: PortfolioConfig,
    risk_off_spy: float,
    assets: List[str],
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Risk-on은 base 그대로 사용.
    Risk-off는 (SPY=risk_off_spy, GLD=base, BTC=base) 고정 후,
    나머지를 BIL에 배분하여 합=1 유지.
    """
    assert assets[0] == "SPY", "Assuming asset order starts with SPY"
    assert "BIL" in assets, "Assuming BIL present"
    assert "GLD" in assets, "Assuming GLD present"

    # risk-on
    w_on = np.array(base_pcfg.w_risk_on, dtype=float)
    assert np.isfinite(w_on).all()
    assert abs(float(w_on.sum()) - 1.0) < 1e-6

    # risk-off: SPY/GLD/BTC fixed, BIL residual
    spy = float(risk_off_spy)
    gld = float(base_pcfg.risk_off_gld)
    btc = float(base_pcfg.risk_off_btc)

    assert 0.0 <= spy <= 1.0
    assert 0.0 <= gld <= 1.0
    assert 0.0 <= btc <= 1.0

    bil = 1.0 - (spy + gld + btc)
    assert bil >= 0.0, f"Invalid risk-off weights: SPY+GLD+BTC={spy+gld+btc:.3f} > 1"

    # construct in exact asset order
    w_off = np.zeros(len(assets), dtype=float)
    for i, a in enumerate(assets):
        if a == "SPY":
            w_off[i] = spy
        elif a == "BIL":
            w_off[i] = bil
        elif a == "GLD":
            w_off[i] = gld
        elif a == "BTC-USD":
            w_off[i] = btc
        else:
            raise ValueError(f"Unknown asset for weight mapping: {a}")

    assert abs(float(w_off.sum()) - 1.0) < 1e-9
    assert (w_off >= -1e-12).all()

    max_single = 0.90
    if logger is not None:
        if np.any(w_on > max_single) or np.any(w_off > max_single):
            logger.warning(f"Single asset weight exceeds {max_single:.0%}")

        if "BTC-USD" in assets:
            btc_idx = assets.index("BTC-USD")
            if w_on[btc_idx] > 0.30 or w_off[btc_idx] > 0.30:
                logger.warning(
                    f"BTC allocation high: on={w_on[btc_idx]:.1%}, off={w_off[btc_idx]:.1%}"
                )
    return w_on, w_off


# =============================
# Backtest
# =============================
def calc_returns(prices: pd.DataFrame) -> pd.DataFrame:
    _assert_sorted_index(prices)
    rets = prices.pct_change().fillna(0.0)
    _assert_sorted_index(rets)
    return rets


def backtest_regime_allocation(
    prices_w: pd.DataFrame,
    regime: pd.Series,
    w_on: np.ndarray,
    w_off: np.ndarray,
    fee_bps: float,
    slippage_bps: float,
    fee_by_asset: Optional[Dict[str, float]] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    _assert_sorted_index(prices_w)
    _assert_sorted_index(regime.to_frame("regime"))

    idx = prices_w.index.intersection(regime.index).sort_values()
    assert len(idx) > 50

    prices_w2 = prices_w.loc[idx].copy()
    regime2 = regime.loc[idx].copy().astype(np.int8)

    assert prices_w2.shape[0] == regime2.shape[0]
    n_assets = prices_w2.shape[1]
    assert n_assets == w_on.shape[0] == w_off.shape[0]

    assert abs(float(w_on.sum()) - 1.0) < 1e-6
    assert abs(float(w_off.sum()) - 1.0) < 1e-6
    assert fee_bps >= 0.0
    assert slippage_bps >= 0.0

    start_ts = time.time()
    rets = calc_returns(prices_w2)

    W = np.zeros((len(idx), n_assets), dtype=np.float32)
    for t, r in enumerate(regime2.values.astype(int)):
        W[t] = w_on if r == 1 else w_off

    assert W.shape == rets.values.shape

    port_ret_gross = (rets.values * W).sum(axis=1)

    delta_w = np.abs(np.diff(W, axis=0)).astype(np.float32)
    turnover = np.zeros(len(idx), dtype=np.float32)
    turnover[1:] = delta_w.sum(axis=1) / 2.0

    asset_fee_bps = np.full(n_assets, fee_bps + slippage_bps, dtype=float)
    if fee_by_asset:
        for i, a in enumerate(prices_w2.columns):
            if a in fee_by_asset:
                asset_fee_bps[i] = float(fee_by_asset[a]) + slippage_bps
    fee = np.zeros(len(idx), dtype=float)
    fee[1:] = (delta_w * asset_fee_bps).sum(axis=1) / (2.0 * 1e4)
    port_ret = port_ret_gross - fee

    contrib = pd.DataFrame(
        rets.values * W,
        index=idx,
        columns=[f"contrib_{c}" for c in prices_w2.columns],
    )

    out = pd.DataFrame(
        {
            "port_ret_gross": port_ret_gross,
            "fee": fee,
            "port_ret": port_ret,
            "regime": regime2.values.astype(int),
            "turnover": turnover,
        },
        index=idx,
    )
    out = pd.concat([out, contrib], axis=1)
    _assert_sorted_index(out)

    if logger is not None:
        elapsed = time.time() - start_ts
        if elapsed > 5.0:
            logger.warning(f"Backtest took {elapsed:.1f}s - consider optimization")
    return out


# =============================
# Metrics & Diagnostics
# =============================
def equity_curve(r: pd.Series) -> pd.Series:
    _assert_sorted_index(r.to_frame("r"))
    eq = (1.0 + r).cumprod()
    _assert_sorted_index(eq.to_frame("eq"))
    return eq


def max_drawdown(eq: pd.Series) -> float:
    _assert_sorted_index(eq.to_frame("eq"))
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def perf_stats(r: pd.Series, periods_per_year: int = 52) -> Dict[str, float]:
    _assert_sorted_index(r.to_frame("r"))
    r2 = r.dropna()
    assert len(r2) > 10

    eq = equity_curve(r2)
    cagr = float(eq.iloc[-1] ** (periods_per_year / len(r2)) - 1.0)

    vol = float(r2.std(ddof=0) * math.sqrt(periods_per_year))
    sharpe = float((r2.mean() * periods_per_year) / (vol + 1e-12))

    mdd = max_drawdown(eq)
    calmar = float(cagr / (abs(mdd) + 1e-12))

    hit = float((r2 > 0).mean())

    try:
        from scipy import stats as _stats
        t_stat, p_val = _stats.ttest_1samp(r2.values, 0.0, nan_policy="omit")
    except Exception:
        t_stat, p_val = np.nan, np.nan
    downside = float(r2[r2 < 0].std(ddof=0) * math.sqrt(periods_per_year))
    sortino = float((r2.mean() * periods_per_year) / (downside + 1e-12)) if downside > 0 else np.nan
    var_95 = float(r2.quantile(0.05))
    cvar_95 = float(r2[r2 <= var_95].mean()) if np.isfinite(var_95) else np.nan

    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "MaxDD": mdd,
        "Calmar": calmar,
        "HitRate": hit,
        "Periods": float(len(r2)),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "Skewness": float(r2.skew()),
        "Kurtosis": float(r2.kurtosis()),
        "Sortino": float(sortino) if np.isfinite(sortino) else np.nan,
        "VaR_95": var_95,
        "CVaR_95": cvar_95,
        "WorstWeek": float(r2.min()),
        "BestWeek": float(r2.max()),
    }


def drawdown_report(out: pd.DataFrame, asset_cols: List[str]) -> Dict[str, object]:
    _assert_sorted_index(out)
    assert "port_ret" in out.columns
    assert "regime" in out.columns

    r = out["port_ret"]
    eq = equity_curve(r)
    peak = eq.cummax()
    dd = eq / peak - 1.0

    trough_date = dd.idxmin()
    trough_dd = float(dd.loc[trough_date])
    peak_date = eq.loc[:trough_date].idxmax()

    post = eq.loc[trough_date:]
    recov_date = None
    for dt, v in post.items():
        if v >= eq.loc[peak_date] - 1e-12:
            recov_date = dt
            break

    window_pt = out.loc[peak_date:trough_date].copy()
    window_pr = out.loc[peak_date:(trough_date if recov_date is None else recov_date)].copy()

    def _pack(win: pd.DataFrame) -> Dict[str, object]:
        regime_ratio = win["regime"].value_counts(normalize=True).to_dict()
        contrib_cols = [f"contrib_{c}" for c in asset_cols]
        for c in contrib_cols:
            assert c in win.columns
        contrib_sum = {k: float(v) for k, v in win[contrib_cols].sum().to_dict().items()}
        return {
            "len": int(len(win)),
            "regime_ratio": {str(k): float(v) for k, v in regime_ratio.items()},
            "contrib_sum": contrib_sum,
            "fee_sum": float(win["fee"].sum()),
            "turnover_sum": float(win["turnover"].sum()),
            "switches": int((win["regime"].diff().fillna(0).abs() > 0).sum()),
        }

    return {
        "mdd": trough_dd,
        "peak_date": str(pd.to_datetime(peak_date).date()),
        "trough_date": str(pd.to_datetime(trough_date).date()),
        "recovery_date": None if recov_date is None else str(pd.to_datetime(recov_date).date()),
        "peak_to_trough": _pack(window_pt),
        "peak_to_recovery": _pack(window_pr),
    }


# =============================
# CV splits
# =============================
def time_series_splits(n: int, n_splits: int, min_train: int, min_val: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    assert n_splits >= 2
    assert min_train >= 52
    assert min_val >= 8
    assert n > min_train + n_splits * min_val

    remain = n - min_train
    fold_size = remain // n_splits
    assert fold_size >= min_val

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        train_end = min_train + k * fold_size
        val_start = train_end
        val_end = min(train_end + fold_size, n)

        tr = np.arange(0, train_end)
        va = np.arange(val_start, val_end)

        assert tr.size >= min_train
        assert va.size >= min_val
        splits.append((tr, va))

    return splits


# =============================
# Trial Params
# =============================
@dataclass(frozen=True)
class TrialParams:
    zwin: int
    smooth_span: int
    threshold: float

    hysteresis: float
    cooldown_weeks: int

    trend_win: int
    use_trend_slope: int  # 0/1

    crash_enabled: int    # 0/1
    crash_ret: float
    crash_lock_weeks: int

    risk_off_spy: float


# =============================
# Backtest wrapper (one-shot full period)
# =============================
def run_full_backtest(
    fred_w: pd.DataFrame,
    prices_w: pd.DataFrame,
    assets: List[str],
    base_fcfg: FeatureConfig,
    base_scfg: BaseSignalConfig,
    base_tcfg: BaseTrendConfig,
    base_ccfg: BaseCrashConfig,
    base_pcfg: PortfolioConfig,
    params: TrialParams,
    weights: Optional[Dict[str, float]] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    _assert_sorted_index(fred_w)
    _assert_sorted_index(prices_w)

    fcfg = FeatureConfig(
        freq=base_fcfg.freq,
        transform=base_fcfg.transform,
        zwin=int(params.zwin),
        smooth_span=int(params.smooth_span),
        clip_z=base_fcfg.clip_z,
        ffill_limit=base_fcfg.ffill_limit,
    )

    scfg = SignalConfig(
        threshold=float(params.threshold),
        hysteresis=float(params.hysteresis),
        cooldown_weeks=int(params.cooldown_weeks),
        adaptive=bool(base_scfg.adaptive),
        adaptive_lookback=int(base_scfg.adaptive_lookback),
        adaptive_k=float(base_scfg.adaptive_k),
    )

    lci_components = None
    if base_fcfg.use_multi_timeframe:
        lci_components = build_multi_timeframe_lci(fred_w, base_fcfg, weights=weights)
        agg = str(base_fcfg.multi_agg)
        if agg == "mean":
            lci = lci_components.mean(axis=1)
        elif agg == "median":
            lci = lci_components.median(axis=1)
        else:
            raise ValueError(f"Unsupported multi_agg: {agg}")
        lci = lci.rename("LCI")
    else:
        lci = build_lci(fred_w, fcfg, weights=weights)
    regime = regime_from_lci(lci, scfg)

    # trend filter
    regime = apply_trend_filter(
        regime=regime,
        prices_w=prices_w,
        tcfg=base_tcfg,
        trend_win=int(params.trend_win),
        use_slope_filter=bool(int(params.use_trend_slope)),
    )

    # crash lockout
    regime = apply_crash_lockout(
        regime=regime,
        prices_w=prices_w,
        enabled=bool(int(params.crash_enabled)),
        crash_ret=float(params.crash_ret),
        lock_weeks=int(params.crash_lock_weeks),
        tkr=base_tcfg.trend_ticker,
        consecutive_threshold=int(base_ccfg.consecutive_threshold),
    )

    # portfolio weights
    w_on, w_off = build_portfolio_weights(
        base_pcfg=base_pcfg,
        risk_off_spy=float(params.risk_off_spy),
        assets=assets,
        logger=logger,
    )

    out = backtest_regime_allocation(
        prices_w=prices_w,
        regime=regime,
        w_on=w_on,
        w_off=w_off,
        fee_bps=float(base_pcfg.fee_bps),
        slippage_bps=float(base_pcfg.slippage_bps),
        fee_by_asset=base_pcfg.fee_by_asset,
        logger=logger,
    )

    out["LCI"] = lci.reindex(out.index).values
    if lci_components is not None:
        for c in lci_components.columns:
            out[c] = lci_components.reindex(out.index)[c].values
    _assert_sorted_index(out)
    return out


def _evaluate_trial(
    tp: TrialParams,
    trial_id: int,
    fred_w: pd.DataFrame,
    prices_w: pd.DataFrame,
    assets: List[str],
    fcfg: FeatureConfig,
    base_scfg: BaseSignalConfig,
    base_tcfg: BaseTrendConfig,
    base_ccfg: BaseCrashConfig,
    base_pcfg: PortfolioConfig,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    idx: pd.DatetimeIndex,
    cvcfg: CVConfig,
    dcfg: DataConfig,
    lci_weights: Optional[Dict[str, float]],
) -> Dict[str, float]:
    out_full = run_full_backtest(
        fred_w=fred_w,
        prices_w=prices_w,
        assets=assets,
        base_fcfg=fcfg,
        base_scfg=base_scfg,
        base_tcfg=base_tcfg,
        base_ccfg=base_ccfg,
        base_pcfg=base_pcfg,
        params=tp,
        weights=lci_weights,
        logger=None,
    )

    fold_stats: List[Dict[str, float]] = []
    for k, (_tr_idx, va_idx) in enumerate(splits):
        va_dates = idx[va_idx]
        out_val = out_full.loc[va_dates].copy()
        assert len(out_val) >= cvcfg.min_val_weeks

        st = perf_stats(out_val["port_ret"])
        st["SwitchesPerYear"] = float(switches_per_year(out_val["regime"]))
        st["fold"] = float(k)
        fold_stats.append(st)

        ck = {
            "trial": asdict(tp),
            "fold": int(k),
            "val_start": str(va_dates[0].date()),
            "val_end": str(va_dates[-1].date()),
            "stats": st,
        }
        ck_path = os.path.join(dcfg.checkpoint_dir, f"trial_{trial_id:05d}", f"fold_{k}.pkl")
        _save_pickle(ck, ck_path)

    df_fs = pd.DataFrame(fold_stats)
    assert len(df_fs) == len(splits)

    mean_metrics = df_fs.drop(columns=["fold"]).mean(numeric_only=True).to_dict()
    mean_metrics = {k: float(v) for k, v in mean_metrics.items()}

    worst_sharpe = float(df_fs["Sharpe"].min())
    min_calmar = float(df_fs["Calmar"].min())
    worst_maxdd = float(df_fs["MaxDD"].min())
    mean_switch = float(df_fs["SwitchesPerYear"].mean())

    score = (
        cvcfg.w_sharpe * worst_sharpe
        + cvcfg.w_calmar * min_calmar
        - cvcfg.w_switch * mean_switch
        - cvcfg.w_worst_dd * abs(worst_maxdd)
    )
    assert np.isfinite(score)

    row: Dict[str, float] = {
        "CAGR": float(mean_metrics.get("CAGR", np.nan)),
        "Vol": float(mean_metrics.get("Vol", np.nan)),
        "Sharpe": float(mean_metrics.get("Sharpe", np.nan)),
        "MaxDD": float(mean_metrics.get("MaxDD", np.nan)),
        "Calmar": float(mean_metrics.get("Calmar", np.nan)),
        "HitRate": float(mean_metrics.get("HitRate", np.nan)),
        "Periods": float(mean_metrics.get("Periods", np.nan)),
        "SwitchesPerYear": mean_switch,
        "WorstSharpe": worst_sharpe,
        "MinCalmar": min_calmar,
        "WorstMaxDD": worst_maxdd,
        "Score": float(score),
        "trial_id": float(trial_id),
        "param_zwin": float(tp.zwin),
        "param_smooth_span": float(tp.smooth_span),
        "param_threshold": float(tp.threshold),
        "param_hysteresis": float(tp.hysteresis),
        "param_cooldown_weeks": float(tp.cooldown_weeks),
        "param_trend_win": float(tp.trend_win),
        "param_use_trend_slope": float(tp.use_trend_slope),
        "param_crash_enabled": float(tp.crash_enabled),
        "param_crash_ret": float(tp.crash_ret),
        "param_crash_lock_weeks": float(tp.crash_lock_weeks),
        "param_risk_off_spy": float(tp.risk_off_spy),
    }
    return row


# =============================
# CV grid search (worst-fold aware)
# =============================
def grid_search_cv(
    fred_w: pd.DataFrame,
    prices_w: pd.DataFrame,
    dcfg: DataConfig,
    fcfg: FeatureConfig,
    base_scfg: BaseSignalConfig,
    base_tcfg: BaseTrendConfig,
    base_ccfg: BaseCrashConfig,
    base_pcfg: PortfolioConfig,
    cvcfg: CVConfig,
    spcfg: SearchSpaceConfig,
    logger: logging.Logger,
    lci_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    _assert_sorted_index(fred_w)
    _assert_sorted_index(prices_w)

    assets = list(prices_w.columns)
    assert assets == list(dcfg.assets), "prices_w column order must match DataConfig.assets"

    idx = prices_w.index.intersection(fred_w.index).sort_values()
    assert len(idx) == len(prices_w), "prices_w must already be aligned to fred_w"
    assert len(idx) > (cvcfg.min_train_weeks + cvcfg.n_splits * cvcfg.min_val_weeks)

    splits = time_series_splits(
        n=len(idx),
        n_splits=cvcfg.n_splits,
        min_train=cvcfg.min_train_weeks,
        min_val=cvcfg.min_val_weeks,
    )

    # build trials (grid)
    trials: List[TrialParams] = []
    for zwin in spcfg.zwin_grid:
        for smooth in spcfg.smooth_grid:
            for thr in spcfg.threshold_grid:
                for h in spcfg.hysteresis_grid:
                    for cd in spcfg.cooldown_grid:
                        for tw in spcfg.trend_win_grid:
                            for ts in spcfg.trend_slope_grid:
                                for ce in spcfg.crash_enable_grid:
                                    # crash disabled면 ret/lock 의미 없지만, 구현 단순화를 위해 그대로 둠
                                    for cr in spcfg.crash_ret_grid:
                                        for cl in spcfg.crash_lock_grid:
                                            for ro_spy in spcfg.risk_off_spy_grid:
                                                trials.append(
                                                    TrialParams(
                                                        zwin=int(zwin),
                                                        smooth_span=int(smooth),
                                                        threshold=float(thr),
                                                        hysteresis=float(h),
                                                        cooldown_weeks=int(cd),
                                                        trend_win=int(tw),
                                                        use_trend_slope=int(ts),
                                                        crash_enabled=int(ce),
                                                        crash_ret=float(cr),
                                                        crash_lock_weeks=int(cl),
                                                        risk_off_spy=float(ro_spy),
                                                    )
                                                )

    logger.info(f"Total trials: {len(trials)} | CV splits: {len(splits)}")
    if cvcfg.use_process_pool and cvcfg.max_workers > 1:
        logger.warning("ProcessPoolExecutor may be memory-heavy due to data serialization")

    rows: List[Dict[str, float]] = []
    batch_size = max(1, int(cvcfg.batch_size))
    max_workers = max(1, int(cvcfg.max_workers))
    use_process_pool = bool(cvcfg.use_process_pool)
    executor_cls = ProcessPoolExecutor if use_process_pool else ThreadPoolExecutor

    for batch_start in range(0, len(trials), batch_size):
        batch = trials[batch_start:batch_start + batch_size]
        if max_workers > 1:
            with executor_cls(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(
                        _evaluate_trial,
                        tp,
                        batch_start + i,
                        fred_w,
                        prices_w,
                        assets,
                        fcfg,
                        base_scfg,
                        base_tcfg,
                        base_ccfg,
                        base_pcfg,
                        splits,
                        idx,
                        cvcfg,
                        dcfg,
                        lci_weights,
                    ): batch_start + i
                    for i, tp in enumerate(batch)
                }
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Grid batch {batch_start // batch_size + 1}",
                ):
                    rows.append(fut.result())
        else:
            for i, tp in enumerate(tqdm(batch, desc=f"Grid batch {batch_start // batch_size + 1}")):
                rows.append(
                    _evaluate_trial(
                        tp,
                        batch_start + i,
                        fred_w,
                        prices_w,
                        assets,
                        fcfg,
                        base_scfg,
                        base_tcfg,
                        base_ccfg,
                        base_pcfg,
                        splits,
                        idx,
                        cvcfg,
                        dcfg,
                        lci_weights,
                    )
                )

    res = pd.DataFrame(rows)
    assert len(res) == len(trials)

    sort_col = str(cvcfg.sort_by)
    assert sort_col in res.columns, f"sort_by not found: {sort_col}"
    res = res.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    return res


def resume_from_checkpoint(dcfg: DataConfig, logger: logging.Logger) -> Optional[pd.DataFrame]:
    cv_path = os.path.join(dcfg.checkpoint_dir, "cv_results.csv")
    if not os.path.exists(cv_path):
        return None
    logger.info(f"Found existing CV results: {cv_path}")
    response = input("Resume from checkpoint? (y/n): ").strip().lower()
    if response != "y":
        return None
    df = pd.read_csv(cv_path)
    logger.warning("Checkpoint resume loads existing results only; missing trials are not re-run.")
    logger.info(f"Loaded {len(df)} completed trials")
    return df


def staged_grid_search(
    fred_w: pd.DataFrame,
    prices_w: pd.DataFrame,
    dcfg: DataConfig,
    fcfg: FeatureConfig,
    base_scfg: BaseSignalConfig,
    base_tcfg: BaseTrendConfig,
    base_ccfg: BaseCrashConfig,
    base_pcfg: PortfolioConfig,
    cvcfg: CVConfig,
    spcfg: SearchSpaceConfig,
    logger: logging.Logger,
    lci_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    logger.info("Staged grid search: Phase 1 (zwin/smooth/threshold)")
    sp1 = SearchSpaceConfig(
        zwin_grid=spcfg.zwin_grid,
        smooth_grid=spcfg.smooth_grid,
        threshold_grid=spcfg.threshold_grid,
        hysteresis_grid=(float(base_scfg.hysteresis),),
        cooldown_grid=(int(base_scfg.cooldown_weeks),),
        trend_win_grid=(int(base_tcfg.trend_win),),
        trend_slope_grid=(int(base_tcfg.use_slope_filter),),
        crash_enable_grid=(int(base_ccfg.enabled),),
        crash_ret_grid=(float(base_ccfg.crash_ret),),
        crash_lock_grid=(int(base_ccfg.lock_weeks),),
        risk_off_spy_grid=(float(base_pcfg.risk_off_spy),),
    )
    dcfg1 = DataConfig(
        start=dcfg.start,
        end=dcfg.end,
        cache_dir=dcfg.cache_dir,
        checkpoint_dir=os.path.join(dcfg.checkpoint_dir, "stage1"),
        log_dir=dcfg.log_dir,
        fred_indicators=dcfg.fred_indicators,
        assets=dcfg.assets,
    )
    res1 = grid_search_cv(
        fred_w=fred_w,
        prices_w=prices_w,
        dcfg=dcfg1,
        fcfg=fcfg,
        base_scfg=base_scfg,
        base_tcfg=base_tcfg,
        base_ccfg=base_ccfg,
        base_pcfg=base_pcfg,
        cvcfg=cvcfg,
        spcfg=sp1,
        logger=logger,
        lci_weights=lci_weights,
    )
    best1 = res1.iloc[0].to_dict()

    logger.info("Staged grid search: Phase 2 (hysteresis/cooldown)")
    sp2 = SearchSpaceConfig(
        zwin_grid=(int(best1["param_zwin"]),),
        smooth_grid=(int(best1["param_smooth_span"]),),
        threshold_grid=(float(best1["param_threshold"]),),
        hysteresis_grid=spcfg.hysteresis_grid,
        cooldown_grid=spcfg.cooldown_grid,
        trend_win_grid=(int(base_tcfg.trend_win),),
        trend_slope_grid=(int(base_tcfg.use_slope_filter),),
        crash_enable_grid=(int(base_ccfg.enabled),),
        crash_ret_grid=(float(base_ccfg.crash_ret),),
        crash_lock_grid=(int(base_ccfg.lock_weeks),),
        risk_off_spy_grid=(float(base_pcfg.risk_off_spy),),
    )
    dcfg2 = DataConfig(
        start=dcfg.start,
        end=dcfg.end,
        cache_dir=dcfg.cache_dir,
        checkpoint_dir=os.path.join(dcfg.checkpoint_dir, "stage2"),
        log_dir=dcfg.log_dir,
        fred_indicators=dcfg.fred_indicators,
        assets=dcfg.assets,
    )
    res2 = grid_search_cv(
        fred_w=fred_w,
        prices_w=prices_w,
        dcfg=dcfg2,
        fcfg=fcfg,
        base_scfg=base_scfg,
        base_tcfg=base_tcfg,
        base_ccfg=base_ccfg,
        base_pcfg=base_pcfg,
        cvcfg=cvcfg,
        spcfg=sp2,
        logger=logger,
        lci_weights=lci_weights,
    )
    best2 = res2.iloc[0].to_dict()

    logger.info("Staged grid search: Phase 3 (trend/crash/risk-off)")
    sp3 = SearchSpaceConfig(
        zwin_grid=(int(best2["param_zwin"]),),
        smooth_grid=(int(best2["param_smooth_span"]),),
        threshold_grid=(float(best2["param_threshold"]),),
        hysteresis_grid=(float(best2["param_hysteresis"]),),
        cooldown_grid=(int(best2["param_cooldown_weeks"]),),
        trend_win_grid=spcfg.trend_win_grid,
        trend_slope_grid=spcfg.trend_slope_grid,
        crash_enable_grid=spcfg.crash_enable_grid,
        crash_ret_grid=spcfg.crash_ret_grid,
        crash_lock_grid=spcfg.crash_lock_grid,
        risk_off_spy_grid=spcfg.risk_off_spy_grid,
    )
    dcfg3 = DataConfig(
        start=dcfg.start,
        end=dcfg.end,
        cache_dir=dcfg.cache_dir,
        checkpoint_dir=os.path.join(dcfg.checkpoint_dir, "stage3"),
        log_dir=dcfg.log_dir,
        fred_indicators=dcfg.fred_indicators,
        assets=dcfg.assets,
    )
    res3 = grid_search_cv(
        fred_w=fred_w,
        prices_w=prices_w,
        dcfg=dcfg3,
        fcfg=fcfg,
        base_scfg=base_scfg,
        base_tcfg=base_tcfg,
        base_ccfg=base_ccfg,
        base_pcfg=base_pcfg,
        cvcfg=cvcfg,
        spcfg=sp3,
        logger=logger,
        lci_weights=lci_weights,
    )
    return res3


def walk_forward_optimization(
    fred_w: pd.DataFrame,
    prices_w: pd.DataFrame,
    dcfg: DataConfig,
    fcfg: FeatureConfig,
    base_scfg: BaseSignalConfig,
    base_tcfg: BaseTrendConfig,
    base_ccfg: BaseCrashConfig,
    base_pcfg: PortfolioConfig,
    cvcfg: CVConfig,
    spcfg: SearchSpaceConfig,
    refit_frequency: int = 52,
    logger: Optional[logging.Logger] = None,
    lci_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    _assert_sorted_index(fred_w)
    _assert_sorted_index(prices_w)
    assert refit_frequency >= 4

    idx = prices_w.index.intersection(fred_w.index).sort_values()
    assert len(idx) == len(prices_w), "prices_w must already be aligned to fred_w"

    out_parts: List[pd.DataFrame] = []
    start = int(cvcfg.min_train_weeks)

    while start + int(cvcfg.min_val_weeks) <= len(idx):
        train_end = start
        test_end = min(train_end + int(refit_frequency), len(idx))
        test_idx = idx[train_end:test_end]
        if len(test_idx) < cvcfg.min_val_weeks:
            break

        if logger is not None:
            logger.info(f"Walk-forward refit at {idx[train_end - 1].date()} (train_end={train_end})")

        fred_train = fred_w.loc[idx[:train_end]].copy()
        prices_train = prices_w.loc[idx[:train_end]].copy()

        cv_res = grid_search_cv(
            fred_w=fred_train,
            prices_w=prices_train,
            dcfg=dcfg,
            fcfg=fcfg,
            base_scfg=base_scfg,
            base_tcfg=base_tcfg,
            base_ccfg=base_ccfg,
            base_pcfg=base_pcfg,
            cvcfg=cvcfg,
            spcfg=spcfg,
            logger=logger or setup_logger(dcfg.log_dir),
            lci_weights=lci_weights,
        )

        best = cv_res.iloc[0].to_dict()
        best_params = TrialParams(
            zwin=int(best["param_zwin"]),
            smooth_span=int(best["param_smooth_span"]),
            threshold=float(best["param_threshold"]),
            hysteresis=float(best["param_hysteresis"]),
            cooldown_weeks=int(best["param_cooldown_weeks"]),
            trend_win=int(best["param_trend_win"]),
            use_trend_slope=int(best["param_use_trend_slope"]),
            crash_enabled=int(best["param_crash_enabled"]),
            crash_ret=float(best["param_crash_ret"]),
            crash_lock_weeks=int(best["param_crash_lock_weeks"]),
            risk_off_spy=float(best["param_risk_off_spy"]),
        )

        out_full = run_full_backtest(
            fred_w=fred_w,
            prices_w=prices_w,
            assets=list(prices_w.columns),
            base_fcfg=fcfg,
            base_scfg=base_scfg,
            base_tcfg=base_tcfg,
            base_ccfg=base_ccfg,
            base_pcfg=base_pcfg,
            params=best_params,
            weights=lci_weights,
            logger=logger,
        )
        out_slice = out_full.loc[test_idx].copy()
        out_slice["refit_date"] = pd.to_datetime(idx[train_end - 1])
        out_parts.append(out_slice)

        start = test_end

    if not out_parts:
        raise RuntimeError("Walk-forward produced no windows")

    out = pd.concat(out_parts, axis=0)
    _assert_sorted_index(out)
    return out


def monitor_strategy_health(out: pd.DataFrame) -> Dict[str, bool]:
    _assert_sorted_index(out)
    assert "regime" in out.columns
    assert "port_ret" in out.columns

    recent = out.tail(52)
    checks = {
        "excessive_switches": float(recent["regime"].diff().abs().sum()) < 10.0,
        "drawdown_acceptable": max_drawdown(equity_curve(recent["port_ret"])) > -0.15,
        "positive_returns": float(recent["port_ret"].sum()) > 0.0,
    }
    return checks


# =============================
# Notifications (Telegram)
# =============================
def send_telegram_message(
    message: str,
    logger: logging.Logger,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> None:
    token = token or os.getenv("TELEGRAM_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("Telegram credentials missing; skipping notification")
        return

    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": message})
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = payload.encode("utf-8")
    try:
        with urllib.request.urlopen(url, data=data, timeout=10) as resp:
            if resp.status != 200:
                logger.warning(f"Telegram notify failed with status {resp.status}")
    except Exception as e:
        logger.warning(f"Telegram notify failed: {e}")


def _rule_enabled(cfg: Dict[str, object], name: str) -> bool:
    rules = cfg.get("rules", {})
    rule = rules.get(name, {})
    return bool(rule.get("enabled", True))


def _rule_value(cfg: Dict[str, object], name: str, key: str, default: object) -> object:
    rules = cfg.get("rules", {})
    rule = rules.get(name, {})
    return rule.get(key, default)


def load_previous_signal(path: str) -> Optional[Dict[str, object]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_current_signal(path: str, signal: Dict[str, object]) -> None:
    _safe_mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(signal, f, ensure_ascii=False, indent=2)


def notify_strategy_events(
    out: pd.DataFrame,
    prices_w: pd.DataFrame,
    prev_regime: Optional[int],
    params: Optional[TrialParams],
    base_ccfg: BaseCrashConfig,
    notif_cfg: Dict[str, object],
    logger: logging.Logger,
) -> None:
    if not notif_cfg.get("enabled", False):
        return

    current_regime = int(out["regime"].iloc[-1])
    current_lci = float(out["LCI"].iloc[-1])
    current_date = pd.to_datetime(out.index[-1]).strftime("%Y-%m-%d")
    current_stats = perf_stats(out.tail(52)["port_ret"])
    health = monitor_strategy_health(out)

    # 1) Regime change
    if prev_regime is not None and prev_regime != current_regime and _rule_enabled(notif_cfg, "regime_change"):
        prev_name = "RISK_ON" if prev_regime == 1 else "RISK_OFF"
        curr_name = "RISK_ON" if current_regime == 1 else "RISK_OFF"
        action = "Increase equity exposure" if current_regime == 1 else "Reduce equity exposure"
        message = (
            "🔔 REGIME CHANGE ALERT\n\n"
            f"Previous: {prev_name}\n"
            f"Current: {curr_name}\n\n"
            f"LCI: {current_lci:.3f}\n"
            f"Date: {current_date}\n"
            f"Recommendation: {action}\n\n"
            f"Recent Performance:\n- Sharpe: {current_stats['Sharpe']:.2f}\n"
            f"- MaxDD: {current_stats['MaxDD']:.1%}"
        )
        send_telegram_message(message, logger)
        logger.info(f"Regime change notification sent: {prev_name} -> {curr_name}")

    # 2) Health warning (weekly or when issues exist)
    if _rule_enabled(notif_cfg, "health_warning"):
        frequency = str(_rule_value(notif_cfg, "health_warning", "frequency", "weekly"))
        if frequency == "weekly" and pd.Timestamp.now().weekday() != 4:
            pass
        else:
            alerts = []
            if not health["excessive_switches"]:
                switches = float(out.tail(52)["regime"].diff().abs().sum())
                alerts.append(f"Excessive switches: {switches:.0f} in last year")
            if not health["drawdown_acceptable"]:
                recent_dd = max_drawdown(equity_curve(out.tail(52)["port_ret"]))
                alerts.append(f"Recent drawdown: {recent_dd:.1%}")
            if current_stats["Sharpe"] < 0.5:
                alerts.append(f"Low Sharpe: {current_stats['Sharpe']:.2f}")
            if alerts:
                message = "⚠️ STRATEGY HEALTH ALERT\n\n" + "\n".join(f"- {a}" for a in alerts)
                send_telegram_message(message, logger)
                logger.warning(f"Health alert sent: {len(alerts)} issues")

    # 3) Large loss alert
    if _rule_enabled(notif_cfg, "large_loss"):
        threshold = float(_rule_value(notif_cfg, "large_loss", "threshold", -0.05))
        recent_return = float(out["port_ret"].iloc[-1])
        if recent_return <= threshold:
            message = (
                "🚨 LARGE LOSS ALERT\n\n"
                f"Today's Loss: {recent_return:.2%}\n"
                f"Date: {current_date}\n\n"
                f"Regime: {'RISK_ON' if current_regime == 1 else 'RISK_OFF'}\n"
                f"LCI: {current_lci:.3f}"
            )
            send_telegram_message(message, logger)
            logger.error(f"Large loss alert sent: {recent_return:.2%}")

    # 4) Crash detection alert
    if _rule_enabled(notif_cfg, "crash_detection"):
        tkr = "SPY"
        if tkr in prices_w.columns:
            recent_ret = float(prices_w[tkr].pct_change().iloc[-1])
            recent_neg = (prices_w[tkr].pct_change() < 0).tail(int(base_ccfg.consecutive_threshold)).sum()
            crash_by_drop = recent_ret <= float(base_ccfg.crash_ret)
            crash_by_streak = base_ccfg.consecutive_threshold > 1 and recent_neg >= base_ccfg.consecutive_threshold
            if crash_by_drop or crash_by_streak:
                message = (
                    "🚨 CRASH DETECTION ALERT\n\n"
                    f"{tkr} daily return: {recent_ret:.2%}\n"
                    f"Crash lockout: {base_ccfg.lock_weeks} weeks\n"
                    f"Date: {current_date}\n\n"
                    "Forced to RISK_OFF mode."
                )
                send_telegram_message(message, logger)
                logger.warning("Crash detection alert sent")

    # 5) Monthly summary
    if _rule_enabled(notif_cfg, "monthly_summary") and out.index[-1].is_month_end:
        monthly_ret = float(out.tail(20)["port_ret"].sum())
        message = (
            f"📊 MONTHLY SUMMARY - {pd.to_datetime(out.index[-1]).strftime('%Y-%m')}\n\n"
            f"Monthly Return: {monthly_ret:.2%}\n"
            f"YTD Return: {float(out['port_ret'].sum()):.2%}\n"
            f"Sharpe (12M): {current_stats['Sharpe']:.2f}\n"
            f"MaxDD (12M): {current_stats['MaxDD']:.1%}\n\n"
            f"Regime: {'RISK_ON' if current_regime == 1 else 'RISK_OFF'}\n"
        )
        if params is not None:
            message += f"Params: zwin={params.zwin}, threshold={params.threshold:.2f}"
        send_telegram_message(message, logger)
        logger.info("Monthly summary sent")


def notify_optimization_complete(
    best_params: TrialParams,
    best_stats: Dict[str, float],
    notif_cfg: Dict[str, object],
    logger: logging.Logger,
) -> None:
    if not notif_cfg.get("enabled", False) or not _rule_enabled(notif_cfg, "optimization_complete"):
        return
    message = (
        "✅ OPTIMIZATION COMPLETE\n\n"
        f"Best Params: zwin={best_params.zwin}, threshold={best_params.threshold:.2f}\n"
        f"Sharpe: {best_stats.get('Sharpe', float('nan')):.2f}\n"
        f"MaxDD: {best_stats.get('MaxDD', float('nan')):.2%}"
    )
    send_telegram_message(message, logger)


def notify_error(
    err: Exception,
    notif_cfg: Dict[str, object],
    logger: logging.Logger,
) -> None:
    if not notif_cfg.get("enabled", False) or not _rule_enabled(notif_cfg, "system_error"):
        return
    message = (
        "❌ PRODUCTION ERROR\n\n"
        f"Error: {err}\n\n"
        f"{traceback.format_exc(limit=3)}"
    )
    send_telegram_message(message, logger)


def live_signal_generator(
    dcfg: DataConfig,
    fcfg: FeatureConfig,
    base_scfg: BaseSignalConfig,
    base_tcfg: BaseTrendConfig,
    base_ccfg: BaseCrashConfig,
    base_pcfg: PortfolioConfig,
    best_params: TrialParams,
    logger: logging.Logger,
    lci_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    out, _prices_a, signal = generate_live_state(
        dcfg=dcfg,
        fcfg=fcfg,
        base_scfg=base_scfg,
        base_tcfg=base_tcfg,
        base_ccfg=base_ccfg,
        base_pcfg=base_pcfg,
        best_params=best_params,
        logger=logger,
        lci_weights=lci_weights,
    )
    return signal


def generate_live_state(
    dcfg: DataConfig,
    fcfg: FeatureConfig,
    base_scfg: BaseSignalConfig,
    base_tcfg: BaseTrendConfig,
    base_ccfg: BaseCrashConfig,
    base_pcfg: PortfolioConfig,
    best_params: TrialParams,
    logger: logging.Logger,
    lci_weights: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    end_date = pd.Timestamp.now().normalize()
    lookback_weeks = max(200, int(best_params.zwin) + 50)
    start_date = end_date - pd.Timedelta(weeks=lookback_weeks)

    fred_key = cache_key(list(dcfg.fred_indicators))
    px_key = cache_key(list(dcfg.assets))

    fred = fetch_fred(
        indicators=list(dcfg.fred_indicators),
        start=str(start_date.date()),
        end=str(end_date.date()),
        cache_path=os.path.join(dcfg.cache_dir, f"fred_live_{fred_key}.parquet"),
        logger=logger,
    )
    prices = fetch_prices_yf(
        tickers=list(dcfg.assets),
        start=str(start_date.date()),
        end=str(end_date.date()),
        cache_path=os.path.join(dcfg.cache_dir, f"prices_live_{px_key}.parquet"),
        logger=logger,
    )

    fred_w = resample_to_freq(fred, fcfg.freq)
    prices_w = resample_to_freq(prices, fcfg.freq)
    fred_a, prices_a = align_data(
        fred_w=fred_w,
        prices_w=prices_w,
        ffill_limit_fred=fcfg.ffill_limit,
        min_non_na_assets=2,
        min_len=max(80, best_params.zwin + best_params.smooth_span + 20),
        logger=logger,
    )

    out = run_full_backtest(
        fred_w=fred_a,
        prices_w=prices_a,
        assets=list(prices_a.columns),
        base_fcfg=fcfg,
        base_scfg=base_scfg,
        base_tcfg=base_tcfg,
        base_ccfg=base_ccfg,
        base_pcfg=base_pcfg,
        params=best_params,
        weights=lci_weights,
        logger=None,
    )
    current_lci = float(out["LCI"].iloc[-1])
    current_regime = int(out["regime"].iloc[-1])
    signal = {
        "date": str(pd.to_datetime(out.index[-1]).date()),
        "lci": current_lci,
        "regime": "RISK_ON" if current_regime == 1 else "RISK_OFF",
        "recommendation": "Increase equity exposure" if current_regime == 1 else "Defensive positioning",
    }
    return out, prices_a, signal


def load_config(config_path: str) -> Dict[str, object]:
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("pyyaml 필요: pip install pyyaml") from e
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def profile_backtest(run_func, logger: logging.Logger) -> None:
    import cProfile
    import pstats
    from io import StringIO

    profiler = cProfile.Profile()
    profiler.enable()
    run_func()
    profiler.disable()

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    logger.info("Performance profiling:")
    logger.info(s.getvalue())


def real_time_backtest(
    prices_w: pd.DataFrame,
    regime: pd.Series,
    w_on: np.ndarray,
    w_off: np.ndarray,
    fee_bps: float,
    slippage_bps: float,
    fee_by_asset: Optional[Dict[str, float]] = None,
    delay_weeks: int = 1,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    _assert_sorted_index(prices_w)
    _assert_sorted_index(regime.to_frame("regime"))
    assert delay_weeks >= 0

    regime_delayed = regime.shift(delay_weeks).fillna(0).astype(int)
    out = backtest_regime_allocation(
        prices_w=prices_w,
        regime=regime_delayed,
        w_on=w_on,
        w_off=w_off,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        fee_by_asset=fee_by_asset,
        logger=logger,
    )
    out["regime_delayed"] = regime_delayed.reindex(out.index).values
    if logger is not None:
        orig_sharpe = perf_stats(out["port_ret"])["Sharpe"]
        logger.info(f"Signal delay applied (weeks={delay_weeks}). Sharpe={orig_sharpe:.3f}")
    return out


def monte_carlo_robustness(
    fred_w: pd.DataFrame,
    prices_w: pd.DataFrame,
    base_fcfg: FeatureConfig,
    base_scfg: BaseSignalConfig,
    base_tcfg: BaseTrendConfig,
    base_ccfg: BaseCrashConfig,
    base_pcfg: PortfolioConfig,
    params: TrialParams,
    n_simulations: int = 200,
    noise_level: float = 0.01,
    logger: Optional[logging.Logger] = None,
    lci_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    _assert_sorted_index(fred_w)
    _assert_sorted_index(prices_w)
    assert n_simulations >= 1
    assert noise_level >= 0.0

    rng = np.random.default_rng()
    results: List[Dict[str, float]] = []

    for i in tqdm(range(n_simulations), desc="Monte Carlo"):
        zwin = max(10, int(params.zwin * (1 + rng.normal(0, noise_level))))
        smooth = max(1, int(params.smooth_span * (1 + rng.normal(0, noise_level))))
        threshold = float(params.threshold + rng.normal(0, noise_level))
        hysteresis = max(0.0, float(params.hysteresis * (1 + rng.normal(0, noise_level))))
        risk_off_spy = min(1.0, max(0.0, float(params.risk_off_spy)))

        noisy = TrialParams(
            zwin=zwin,
            smooth_span=smooth,
            threshold=threshold,
            hysteresis=hysteresis,
            cooldown_weeks=params.cooldown_weeks,
            trend_win=params.trend_win,
            use_trend_slope=params.use_trend_slope,
            crash_enabled=params.crash_enabled,
            crash_ret=params.crash_ret,
            crash_lock_weeks=params.crash_lock_weeks,
            risk_off_spy=risk_off_spy,
        )
        try:
            out = run_full_backtest(
                fred_w=fred_w,
                prices_w=prices_w,
                assets=list(prices_w.columns),
                base_fcfg=base_fcfg,
                base_scfg=base_scfg,
                base_tcfg=base_tcfg,
                base_ccfg=base_ccfg,
                base_pcfg=base_pcfg,
                params=noisy,
                weights=lci_weights,
                logger=None,
            )
            stats = perf_stats(out["port_ret"])
            results.append(stats)
        except Exception as e:
            if logger is not None:
                logger.warning(f"MC simulation {i} failed: {e}")

    df = pd.DataFrame(results)
    if logger is not None and not df.empty:
        logger.info(
            f"Monte Carlo results (n={len(df)}): Sharpe={df['Sharpe'].mean():.3f} ± {df['Sharpe'].std():.3f}"
        )
        logger.info(
            f"Monte Carlo results (n={len(df)}): MaxDD={df['MaxDD'].mean():.3f} ± {df['MaxDD'].std():.3f}"
        )
    return df


# =============================
# Smoke Test
# =============================
def smoke_test(logger: logging.Logger) -> None:
    dcfg = DataConfig(start="2018-01-01", end="2020-12-31")

    fcfg = FeatureConfig(zwin=26, smooth_span=4)
    base_scfg = BaseSignalConfig(threshold=0.0, hysteresis=0.2, cooldown_weeks=4)
    base_tcfg = BaseTrendConfig(trend_ticker="SPY", trend_win=20, use_slope_filter=False)
    base_ccfg = BaseCrashConfig(enabled=False, crash_ret=-0.07, lock_weeks=8)
    base_pcfg = PortfolioConfig(risk_off_spy=0.10, risk_off_gld=0.15, risk_off_btc=0.00, fee_bps=2.0)

    fred_key = cache_key(list(dcfg.fred_indicators))
    px_key = cache_key(list(dcfg.assets))

    fred = fetch_fred(
        indicators=list(dcfg.fred_indicators),
        start=dcfg.start,
        end=dcfg.end,
        cache_path=os.path.join(dcfg.cache_dir, f"fred_smoke_{fred_key}.parquet"),
        logger=logger,
    )
    prices = fetch_prices_yf(
        tickers=list(dcfg.assets),
        start=dcfg.start,
        end=dcfg.end,
        cache_path=os.path.join(dcfg.cache_dir, f"prices_smoke_{px_key}.parquet"),
        logger=logger,
    )

    fred_w = resample_to_freq(fred, fcfg.freq)
    prices_w = resample_to_freq(prices, fcfg.freq)

    min_len_smoke = max(80, fcfg.zwin + fcfg.smooth_span + 20)
    fred_a, prices_a = align_data(
        fred_w=fred_w,
        prices_w=prices_w,
        ffill_limit_fred=fcfg.ffill_limit,
        min_non_na_assets=2,
        min_len=min_len_smoke,
        logger=logger,
    )

    tp = TrialParams(
        zwin=fcfg.zwin,
        smooth_span=fcfg.smooth_span,
        threshold=-0.25,
        hysteresis=0.1,
        cooldown_weeks=0,
        trend_win=20,
        use_trend_slope=0,
        crash_enabled=0,
        crash_ret=-0.07,
        crash_lock_weeks=8,
        risk_off_spy=0.10,
    )

    out = run_full_backtest(
        fred_w=fred_a,
        prices_w=prices_a,
        assets=list(prices_a.columns),
        base_fcfg=fcfg,
        base_scfg=base_scfg,
        base_tcfg=base_tcfg,
        base_ccfg=base_ccfg,
        base_pcfg=base_pcfg,
        params=tp,
        weights=None,
        logger=logger,
    )

    stats = perf_stats(out["port_ret"])
    stats["SwitchesPerYear"] = switches_per_year(out["regime"])
    assert np.isfinite(list(stats.values())).all()
    logger.info(f"Smoke test OK. Stats: {stats}")


# =============================
# Main
# =============================
def main() -> None:
    args = parse_args()
    config_dict = load_config(args.config) if args.config else None
    notif_cfg = get_notification_config(config_dict, args)

    dcfg, fcfg, base_scfg, base_tcfg, base_ccfg, base_pcfg, cvcfg, spcfg = build_configs(
        args, config_dict
    )

    logger = setup_logger(dcfg.log_dir)
    _safe_mkdir(dcfg.cache_dir)
    _safe_mkdir(dcfg.checkpoint_dir)
    fallback_params = default_params_from_configs(fcfg, base_scfg, base_tcfg, base_ccfg, base_pcfg)

    if fcfg.use_multi_timeframe:
        logger.info("Multi-timeframe LCI enabled; zwin/smooth grid params are ignored for LCI.")

    def run_backtest_flow(
        optimize_only: bool,
    ) -> Tuple[Optional[pd.DataFrame], Optional[TrialParams], Optional[Dict[str, float]]]:
        smoke_test(logger)
        logger.info("Load data...")
        fred_a, prices_a = load_and_align_data(dcfg, fcfg, logger)

        if args.walk_forward:
            logger.info("Run walk-forward optimization...")
            out_wf = walk_forward_optimization(
                fred_w=fred_a,
                prices_w=prices_a,
                dcfg=dcfg,
                fcfg=fcfg,
                base_scfg=base_scfg,
                base_tcfg=base_tcfg,
                base_ccfg=base_ccfg,
                base_pcfg=base_pcfg,
                cvcfg=cvcfg,
                spcfg=spcfg,
                refit_frequency=52,
                logger=logger,
                lci_weights=None,
            )
            return out_wf, None, None

        resume_df = resume_from_checkpoint(dcfg, logger) if args.resume else None
        if resume_df is not None:
            cv_res = resume_df
        elif cvcfg.use_staged_search:
            logger.info("Run staged CV grid search...")
            cv_res = staged_grid_search(
                fred_w=fred_a,
                prices_w=prices_a,
                dcfg=dcfg,
                fcfg=fcfg,
                base_scfg=base_scfg,
                base_tcfg=base_tcfg,
                base_ccfg=base_ccfg,
                base_pcfg=base_pcfg,
                cvcfg=cvcfg,
                spcfg=spcfg,
                logger=logger,
                lci_weights=None,
            )
        else:
            logger.info("Run CV grid search...")
            cv_res = grid_search_cv(
                fred_w=fred_a,
                prices_w=prices_a,
                dcfg=dcfg,
                fcfg=fcfg,
                base_scfg=base_scfg,
                base_tcfg=base_tcfg,
                base_ccfg=base_ccfg,
                base_pcfg=base_pcfg,
                cvcfg=cvcfg,
                spcfg=spcfg,
                logger=logger,
                lci_weights=None,
            )

        cv_path = os.path.join(dcfg.checkpoint_dir, "cv_results.csv")
        cv_res.to_csv(cv_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved CV results: {cv_path}")

        best = cv_res.iloc[0].to_dict()
        logger.info(f"Best trial (by {cvcfg.sort_by}): {best}")

        best_params = TrialParams(
            zwin=int(best["param_zwin"]),
            smooth_span=int(best["param_smooth_span"]),
            threshold=float(best["param_threshold"]),
            hysteresis=float(best["param_hysteresis"]),
            cooldown_weeks=int(best["param_cooldown_weeks"]),
            trend_win=int(best["param_trend_win"]),
            use_trend_slope=int(best["param_use_trend_slope"]),
            crash_enabled=int(best["param_crash_enabled"]),
            crash_ret=float(best["param_crash_ret"]),
            crash_lock_weeks=int(best["param_crash_lock_weeks"]),
            risk_off_spy=float(best["param_risk_off_spy"]),
        )

        if optimize_only:
            best_stats = {
                "Sharpe": float(best.get("Sharpe", np.nan)),
                "MaxDD": float(best.get("MaxDD", np.nan)),
                "Score": float(best.get("Score", np.nan)),
            }
            return None, best_params, best_stats

        out = run_full_backtest(
            fred_w=fred_a,
            prices_w=prices_a,
            assets=list(prices_a.columns),
            base_fcfg=fcfg,
            base_scfg=base_scfg,
            base_tcfg=base_tcfg,
            base_ccfg=base_ccfg,
            base_pcfg=base_pcfg,
            params=best_params,
            weights=None,
            logger=logger,
        )
        best_stats = {
            "Sharpe": float(best.get("Sharpe", np.nan)),
            "MaxDD": float(best.get("MaxDD", np.nan)),
            "Score": float(best.get("Score", np.nan)),
        }
        return out, best_params, best_stats

    try:
        if args.mode == "profile":
            profile_backtest(lambda: run_backtest_flow(optimize_only=False), logger)
            return

        if args.mode == "live":
            best_params = load_best_params_from_cv(dcfg, fallback_params=fallback_params)
            logger.info("Running live signal generation...")
            out_live, prices_live, signal = generate_live_state(
                dcfg,
                fcfg,
                base_scfg,
                base_tcfg,
                base_ccfg,
                base_pcfg,
                best_params,
                logger,
            )
            print(json.dumps(signal, indent=2))
            state_path = os.path.join(dcfg.checkpoint_dir, "last_signal.json")
            prev_state = load_previous_signal(state_path)
            prev_regime = None
            if prev_state is not None:
                prev_val = prev_state.get("regime_int")
                if prev_val in (0, 1):
                    prev_regime = int(prev_val)
            notify_strategy_events(
                out=out_live,
                prices_w=prices_live,
                prev_regime=prev_regime,
                params=best_params,
                base_ccfg=base_ccfg,
                notif_cfg=notif_cfg,
                logger=logger,
            )
            signal_record = dict(signal)
            signal_record["regime_int"] = 1 if signal["regime"] == "RISK_ON" else 0
            save_current_signal(state_path, signal_record)
            return

        if args.mode == "monte_carlo":
            best_params = load_best_params_from_cv(dcfg, fallback_params=fallback_params)
            logger.info(f"Running Monte Carlo robustness test ({args.mc_sims} sims)...")
            fred_a, prices_a = load_and_align_data(dcfg, fcfg, logger)
            mc_results = monte_carlo_robustness(
                fred_a,
                prices_a,
                fcfg,
                base_scfg,
                base_tcfg,
                base_ccfg,
                base_pcfg,
                best_params,
                n_simulations=args.mc_sims,
                logger=logger,
            )
            mc_path = os.path.join(dcfg.checkpoint_dir, "monte_carlo_results.csv")
            mc_results.to_csv(mc_path, index=False)
            logger.info(f"Saved Monte Carlo results: {mc_path}")
            return

        if args.mode == "optimize":
            _out, best_params, best_stats = run_backtest_flow(optimize_only=True)
            if best_params is not None and best_stats is not None:
                notify_optimization_complete(best_params, best_stats, notif_cfg, logger)
            return

        out, best_params, best_stats = run_backtest_flow(optimize_only=False)
        assert out is not None

        final_stats = perf_stats(out["port_ret"])
        final_stats["SwitchesPerYear"] = switches_per_year(out["regime"])
        logger.info(f"Final stats: {final_stats}")

        out_path = os.path.join(dcfg.checkpoint_dir, "final_backtest.parquet")
        _to_parquet(out, out_path)
        logger.info(f"Saved final backtest: {out_path}")

        stats_path = os.path.join(dcfg.checkpoint_dir, "final_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_params": None if best_params is None else asdict(best_params),
                    "walk_forward": bool(args.walk_forward),
                    "stats": final_stats,
                    "cv_sort_by": cvcfg.sort_by,
                    "cv_score_weights": {
                        "w_sharpe": cvcfg.w_sharpe,
                        "w_calmar": cvcfg.w_calmar,
                        "w_switch": cvcfg.w_switch,
                        "w_worst_dd": cvcfg.w_worst_dd,
                    },
                    "assets": list(dcfg.assets),
                    "fred_indicators": list(dcfg.fred_indicators),
                    "portfolio_config": asdict(base_pcfg),
                    "feature_config": asdict(fcfg),
                    "base_signal_config": asdict(base_scfg),
                    "base_trend_config": asdict(base_tcfg),
                    "base_crash_config": asdict(base_ccfg),
                    "search_space": asdict(spcfg),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(f"Saved final stats: {stats_path}")

        dd_rep = drawdown_report(out, asset_cols=list(dcfg.assets))
        dd_path = os.path.join(dcfg.checkpoint_dir, "final_drawdown_report.json")
        with open(dd_path, "w", encoding="utf-8") as f:
            json.dump(dd_rep, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved drawdown report: {dd_path}")

        if best_params is not None and best_stats is not None:
            notify_optimization_complete(best_params, best_stats, notif_cfg, logger)
    except Exception as e:
        notify_error(e, notif_cfg, logger)
        raise


if __name__ == "__main__":
    main()
