# Copyright (c) HelkiQuant contributors.
"""
analyze_multi_layer.py
======================
对 artifacts/<exp_name>/ 下导出的三层 CatBoost+DEnsemble
模型做全面分析，并生成单一 HTML 报告。

用法：
    python analyze_multi_layer.py run \
        --artifacts_dir artifacts/robust_v2 \
        --target SZ301536 \
        --out artifacts/robust_v2/model_report.html

包含内容：
    1. 配置 / 元信息总览（每层 loss / num_models / 特征数 / 阈值）
    2. 训练标签真值分布（外层 21 日收益分类比例 / 中层 5 日 / 内层 1 日）
    3. 三层 pred 信号在测试期的统计与直方图
    4. 三层 pred IC（截面 + 时序）
    5. 三层 pred 互相关（layer-vs-layer）
    6. 标的（target stock）专属：三层信号时间序列 + 真实收益对照
    7. 三层信号 → MultiLayerStrategy 决策路径计数
    8. CatBoost 子模型特征重要性（每层 top-30）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from catboost import CatBoost

import qlib
from qlib.data import D

DIRNAME = Path(__file__).absolute().resolve().parent
INTRADAY_DIR = DIRNAME.parent / "intraday_t"
if str(DIRNAME) not in sys.path:
    sys.path.insert(0, str(DIRNAME))
if str(INTRADAY_DIR) not in sys.path:
    sys.path.insert(0, str(INTRADAY_DIR))


# ============================================================ #
# 工具
# ============================================================ #
def _read_pred(path: Path, col: str) -> pd.Series:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    s = df.set_index(["datetime", "instrument"])[col]
    return s


def _load_layer_meta(layer_dir: Path) -> dict:
    meta = json.loads((layer_dir / "meta.json").read_text(encoding="utf-8"))
    handler_kwargs = json.loads(
        (layer_dir / "handler_kwargs.json").read_text(encoding="utf-8")
    )
    feature_names = (layer_dir / "feature_names.txt").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    sub_features = []
    for i in range(meta["num_models"]):
        feats = json.loads(
            (layer_dir / f"sub_features_{i}.json").read_text(encoding="utf-8")
        )
        sub_features.append(feats)
    return {
        "meta": meta,
        "handler_kwargs": handler_kwargs,
        "feature_names": feature_names,
        "sub_features": sub_features,
    }


def _load_submodels(layer_dir: Path, num_models: int) -> list[CatBoost]:
    models = []
    for i in range(num_models):
        m = CatBoost()
        m.load_model(str(layer_dir / f"submodel_{i}.cbm"))
        models.append(m)
    return models


def _label_expr_from_handler(handler_kwargs: dict) -> str:
    """从 handler_kwargs['label'] 取出第一个标签表达式。"""
    label = handler_kwargs.get("label", [])
    if isinstance(label, list) and len(label) > 0:
        return label[0]
    return ""


def _fetch_close(target: str, start: str, end: str, freq: str = "day") -> pd.Series:
    """取目标股票收盘价时序。"""
    df = D.features([target], ["$close"], start_time=start, end_time=end, freq=freq)
    if df.empty:
        return pd.Series(dtype=float)
    s = df["$close"]
    if isinstance(s.index, pd.MultiIndex):
        if "instrument" in s.index.names:
            s = s.droplevel("instrument")
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _ic_cs(pred: pd.Series, label: pd.Series) -> pd.Series:
    """逐日截面 IC（spearman 等价于 rank 后 pearson，这里用 pearson on rank）。"""
    df = pd.DataFrame({"p": pred, "y": label}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    def _f(g):
        if len(g) < 5:
            return np.nan
        return g["p"].rank().corr(g["y"].rank())
    return df.groupby(level="datetime").apply(_f)


def _ic_ts(pred: pd.Series, label: pd.Series, instrument: str) -> float:
    """单只股票时序 IC。"""
    if isinstance(pred.index, pd.MultiIndex):
        try:
            p = pred.xs(instrument, level="instrument")
        except KeyError:
            return np.nan
    else:
        p = pred
    if isinstance(label.index, pd.MultiIndex):
        try:
            y = label.xs(instrument, level="instrument")
        except KeyError:
            return np.nan
    else:
        y = label
    df = pd.DataFrame({"p": p, "y": y}).dropna()
    if len(df) < 10:
        return np.nan
    return df["p"].rank().corr(df["y"].rank())


def _bin_classify(y: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """按 thresholds 把连续标签离散化（与 CatBoostDEnsemble 一致）。"""
    bins = [-np.inf] + sorted(thresholds) + [np.inf]
    return np.digitize(y, bins[1:-1]).astype(int)


# ============================================================ #
# 标签真值取数
# ============================================================ #
def _materialize_label(target: str, start: str, end: str, expr: str) -> pd.Series:
    """从 qlib 取目标股票一个标签表达式的值。"""
    df = D.features([target], [expr], start_time=start, end_time=end, freq="day")
    if df.empty:
        return pd.Series(dtype=float)
    s = df.iloc[:, 0]
    if isinstance(s.index, pd.MultiIndex):
        # qlib 默认 (instrument, datetime)
        if "instrument" in s.index.names:
            s = s.droplevel("instrument")
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _materialize_label_pool(instruments: list[str], start: str, end: str,
                            expr: str) -> pd.Series:
    """整池标签 (multi-index: datetime, instrument)。"""
    if not instruments:
        return pd.Series(dtype=float)
    df = D.features(instruments, [expr],
                    start_time=start, end_time=end, freq="day")
    if df.empty:
        return pd.Series(dtype=float)
    s = df.iloc[:, 0]
    # qlib 默认返回 (instrument, datetime)，需要换成 (datetime, instrument)
    if isinstance(s.index, pd.MultiIndex):
        names = list(s.index.names)
        if names == ["instrument", "datetime"]:
            s = s.swaplevel("instrument", "datetime")
        # 把 datetime 那一层强制转成 Timestamp
        new_index = pd.MultiIndex.from_arrays(
            [pd.to_datetime(s.index.get_level_values("datetime")),
             s.index.get_level_values("instrument")],
            names=["datetime", "instrument"],
        )
        s.index = new_index
    return s.sort_index()


# ============================================================ #
# 图表渲染（用 plotly 内联 HTML）
# ============================================================ #
def _try_plotly():
    try:
        import plotly.graph_objects as go  # noqa
        import plotly.io as pio  # noqa
        return True
    except ImportError:
        return False


HAS_PLOTLY = _try_plotly()


def _hist_html(s: pd.Series, title: str, color: str = "#3b82f6") -> str:
    if not HAS_PLOTLY:
        return f"<pre>{title} (plotly missing)\n{s.describe().to_string()}</pre>"
    import plotly.graph_objects as go
    s_clean = s.dropna()
    fig = go.Figure(
        go.Histogram(x=s_clean, nbinsx=80, marker_color=color, opacity=0.8)
    )
    fig.update_layout(
        title=title, height=320, margin=dict(l=40, r=20, t=40, b=40),
        bargap=0.05, paper_bgcolor="#fff", plot_bgcolor="#fafafa",
    )
    return fig.to_html(include_plotlyjs=False, full_html=False)


def _line_html(df: pd.DataFrame, title: str) -> str:
    if not HAS_PLOTLY:
        return f"<pre>{title}\n{df.tail(50).to_string()}</pre>"
    import plotly.graph_objects as go
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], mode="lines", name=col, line=dict(width=1.4)
        ))
    fig.update_layout(
        title=title, height=380, margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="#fff", plot_bgcolor="#fafafa",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig.to_html(include_plotlyjs=False, full_html=False)


def _bar_html(s: pd.Series, title: str, color: str = "#10b981") -> str:
    if not HAS_PLOTLY:
        return f"<pre>{title}\n{s.to_string()}</pre>"
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(x=s.index.astype(str), y=s.values, marker_color=color))
    fig.update_layout(
        title=title, height=360, margin=dict(l=40, r=20, t=40, b=80),
        paper_bgcolor="#fff", plot_bgcolor="#fafafa",
    )
    return fig.to_html(include_plotlyjs=False, full_html=False)


# ============================================================ #
# 决策路径（与 multi_layer_strategy 同步的简化版）
# ============================================================ #
def _decision_path(p_outer: float, p_mid: float, p_inner: float,
                   outer_upper: float, outer_lower: float,
                   mid_buy: float, mid_sell: float,
                   inner_buy: float, inner_sell: float) -> str:
    if not (np.isfinite(p_outer) and np.isfinite(p_mid) and np.isfinite(p_inner)):
        return "NaN"
    if p_outer > outer_upper:
        if p_mid > mid_buy:
            return "A.主升浪.加仓"
        if p_mid < mid_sell:
            return "A.主升浪.减仓"
        return "A.主升浪.观望"
    elif p_outer >= outer_lower:
        if p_mid > mid_buy:
            if p_inner > inner_buy:
                return "B.震荡.买"
            return "B.震荡.买(内层弱)"
        if p_mid < mid_sell:
            if p_inner < inner_sell:
                return "B.震荡.卖"
            return "B.震荡.卖(内层弱)"
        return "B.震荡.观望"
    else:
        if p_mid < mid_sell:
            return "C.主跌浪.高抛"
        if p_mid > mid_buy:
            return "C.主跌浪.低吸"
        return "C.主跌浪.观望"


# ============================================================ #
# 主入口
# ============================================================ #
def run(
    artifacts_dir: str,
    target: str = "SZ301536",
    out: str | None = None,
    provider_uri_day: str = str((DIRNAME / "../../data/cn_data_pool").resolve()),
    provider_uri_1min: str = str((DIRNAME / "../../data/cn_data_1min").resolve()),
    strategy_outer_upper: float = -0.06,
    strategy_outer_lower: float = -0.24,
    strategy_mid_buy: float = 0.01,
    strategy_mid_sell: float = -0.01,
    strategy_inner_buy: float = 0.003,
    strategy_inner_sell: float = -0.003,
    sample_pool_for_label: int = 200,
):
    """生成完整模型分析 HTML 报告。

    Parameters
    ----------
    artifacts_dir : str
        三层模型导出目录（包含 outer/middle/inner/predictions 子目录）
    target : str
        重点关注股票（时间序列分析对象）
    out : str
        输出 HTML 路径；None 则取 artifacts_dir/model_report.html
    sample_pool_for_label : int
        计算池内截面 IC 时随机采样多少只股票（避免数据量过大）
    """
    art_dir = Path(artifacts_dir).expanduser().resolve()
    if not art_dir.exists():
        raise FileNotFoundError(f"artifacts_dir not found: {art_dir}")
    out_path = Path(out) if out else art_dir / "model_report.html"

    print(f"[init] qlib day={provider_uri_day}")
    qlib.init(
        provider_uri={"day": provider_uri_day, "1min": provider_uri_1min},
        dataset_cache=None, expression_cache=None, region="cn",
    )

    # ============================================================ #
    # 1. 载入三层 meta + pred
    # ============================================================ #
    layers = {}
    for name in ["outer", "middle", "inner"]:
        layer_dir = art_dir / name
        if not layer_dir.exists():
            raise FileNotFoundError(f"layer dir missing: {layer_dir}")
        info = _load_layer_meta(layer_dir)
        info["pred"] = _read_pred(art_dir / "predictions" / f"pred_{name}.csv",
                                  f"pred_{name}")
        layers[name] = info
        print(f"[layer:{name}] loss={info['meta']['loss']} "
              f"num_models={info['meta']['num_models']} "
              f"feat={len(info['feature_names'])} "
              f"pred_rows={len(info['pred'])}")

    # ============================================================ #
    # 2. 计算 IC
    # ============================================================ #
    ic_records = []  # 每层 (cross_ic_mean, cross_ic_ir, cross_ic_pos_pct, ts_ic_target)
    for name, info in layers.items():
        pred = info["pred"]
        # 从 handler_kwargs 取标签表达式
        expr = _label_expr_from_handler(info["handler_kwargs"])
        if not expr:
            ic_records.append({"layer": name, "label_expr": "", "n_pred": len(pred)})
            continue

        # pred 索引 datetime 范围
        dts = pred.index.get_level_values("datetime")
        start = pd.to_datetime(dts.min())
        end = pd.to_datetime(dts.max()) + pd.Timedelta(days=30)

        # 池内股票（采样）以截面 IC
        instruments = pred.index.get_level_values("instrument").unique().tolist()
        rng = np.random.default_rng(42)
        if len(instruments) > sample_pool_for_label:
            sampled = rng.choice(instruments, size=sample_pool_for_label, replace=False)
            sampled = list(sampled)
        else:
            sampled = instruments
        if target not in sampled:
            sampled = sampled + [target]

        print(f"[ic:{name}] computing label `{expr}` for {len(sampled)} instruments")
        try:
            label_pool = _materialize_label_pool(
                sampled, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), expr
            )
        except Exception as e:
            print(f"  [warn] label materialize failed: {e}")
            label_pool = pd.Series(dtype=float)

        if not label_pool.empty:
            # 把 pred 限制在采样集
            pred_sub = pred[pred.index.get_level_values("instrument").isin(sampled)]
            cs_ic = _ic_cs(pred_sub, label_pool)
            ts_ic_target = _ic_ts(pred_sub, label_pool, target)
        else:
            cs_ic = pd.Series(dtype=float)
            ts_ic_target = np.nan

        ic_records.append({
            "layer": name, "label_expr": expr,
            "n_pred": len(pred),
            "cs_ic_mean": float(cs_ic.mean()) if not cs_ic.empty else np.nan,
            "cs_ic_std": float(cs_ic.std()) if not cs_ic.empty else np.nan,
            "cs_ic_ir": float(cs_ic.mean() / cs_ic.std()) if not cs_ic.empty and cs_ic.std() > 0 else np.nan,
            "cs_ic_pos_pct": float((cs_ic > 0).mean()) if not cs_ic.empty else np.nan,
            "ts_ic_target": ts_ic_target,
            "cs_ic_series": cs_ic,
        })

    # ============================================================ #
    # 3. 标的时序：三层信号 + 真实收益
    # ============================================================ #
    target_series = {}
    for name, info in layers.items():
        pred = info["pred"]
        if isinstance(pred.index, pd.MultiIndex):
            try:
                s = pred.xs(target, level="instrument")
                target_series[name] = s
            except KeyError:
                pass

    # 真实收盘价 + 1日收益
    if target_series:
        all_dts = sorted(set().union(*[s.index for s in target_series.values()]))
        if all_dts:
            close = _fetch_close(target,
                                 pd.to_datetime(all_dts[0]).strftime("%Y-%m-%d"),
                                 pd.to_datetime(all_dts[-1] + pd.Timedelta(days=30)).strftime("%Y-%m-%d"))
        else:
            close = pd.Series(dtype=float)
    else:
        close = pd.Series(dtype=float)

    # ============================================================ #
    # 4. 决策路径计数（在 target 上）
    # ============================================================ #
    decision_counts = pd.Series(dtype=int)
    if all(k in target_series for k in ["outer", "middle", "inner"]):
        df_t = pd.concat({
            "outer": target_series["outer"],
            "middle": target_series["middle"],
            "inner": target_series["inner"],
        }, axis=1).dropna()
        paths = []
        for _, r in df_t.iterrows():
            paths.append(_decision_path(
                r["outer"], r["middle"], r["inner"],
                strategy_outer_upper, strategy_outer_lower,
                strategy_mid_buy, strategy_mid_sell,
                strategy_inner_buy, strategy_inner_sell,
            ))
        decision_counts = pd.Series(paths).value_counts().sort_values(ascending=False)

    # ============================================================ #
    # 5. 三层互相关 (target 时序)
    # ============================================================ #
    cross_corr_html = "<p><i>insufficient data for cross-corr</i></p>"
    if all(k in target_series for k in ["outer", "middle", "inner"]):
        df_t = pd.concat({
            "outer": target_series["outer"],
            "middle": target_series["middle"],
            "inner": target_series["inner"],
        }, axis=1).dropna()
        if not df_t.empty:
            corr = df_t.corr().round(4)
            cross_corr_html = corr.to_html(classes="tbl", border=0)

    # ============================================================ #
    # 6. 子模型特征重要性
    # ============================================================ #
    importance_blocks = {}
    for name, info in layers.items():
        layer_dir = art_dir / name
        try:
            submodels = _load_submodels(layer_dir, info["meta"]["num_models"])
        except Exception as e:
            importance_blocks[name] = f"<p>load submodels failed: {e}</p>"
            continue
        # 每个子模型用 own sub_features 算重要性，再聚合
        agg = {}
        for i, m in enumerate(submodels):
            try:
                imp = m.get_feature_importance()
            except Exception:
                continue
            feats = info["sub_features"][i]
            if len(imp) != len(feats):
                continue
            for f, v in zip(feats, imp):
                agg[f] = agg.get(f, 0.0) + float(v)
        if not agg:
            importance_blocks[name] = "<p>no importance</p>"
            continue
        ser = pd.Series(agg).sort_values(ascending=False).head(30)
        importance_blocks[name] = _bar_html(ser, f"[{name}] Top-30 特征重要性（多子模型聚合）",
                                             color={"outer": "#ef4444",
                                                    "middle": "#f59e0b",
                                                    "inner": "#3b82f6"}[name])

    # ============================================================ #
    # 7. 标签真值分布（target stock 上的 21d/5d/1d 收益）
    # ============================================================ #
    label_dist_blocks = {}
    label_class_pcts = {}
    for name, info in layers.items():
        expr = _label_expr_from_handler(info["handler_kwargs"])
        if not expr:
            continue
        try:
            pred = info["pred"]
            dts = pred.index.get_level_values("datetime")
            start = pd.to_datetime(dts.min()).strftime("%Y-%m-%d")
            end = (pd.to_datetime(dts.max()) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            label_t = _materialize_label(target, start, end, expr).dropna()
            if label_t.empty:
                continue
            color = {"outer": "#dc2626",
                     "middle": "#d97706",
                     "inner": "#2563eb"}[name]
            label_dist_blocks[name] = _hist_html(
                label_t, f"[{name}] {target} 真实标签分布: {expr}", color=color)
            # 外层分类比例
            if info["meta"]["is_cls"]:
                cls = _bin_classify(label_t.values, info["meta"]["thresholds"])
                pct = pd.Series(cls).value_counts(normalize=True).sort_index()
                names = ["主跌浪", "震荡", "主升浪"][:len(pct)]
                pct.index = [names[i] if i < len(names) else f"cls_{i}" for i in pct.index]
                label_class_pcts[name] = pct
        except Exception as e:
            label_dist_blocks[name] = f"<p>label fetch failed: {e}</p>"

    # ============================================================ #
    # 8. 渲染 HTML
    # ============================================================ #
    def fmt(x, n=4):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "—"
        return f"{x:.{n}f}"

    # 概览表
    overview_rows = []
    for name in ["outer", "middle", "inner"]:
        info = layers[name]
        m = info["meta"]
        overview_rows.append({
            "layer": name,
            "loss": m["loss"],
            "is_cls": m["is_cls"],
            "thresholds": m.get("thresholds"),
            "num_sub_models": m["num_models"],
            "sub_weights": m["sub_weights"],
            "n_features_total": len(info["feature_names"]),
            "n_pred_rows": len(info["pred"]),
            "label_expr": _label_expr_from_handler(info["handler_kwargs"]),
        })
    overview_df = pd.DataFrame(overview_rows).set_index("layer")
    overview_html = overview_df.to_html(classes="tbl", border=0)

    # IC 表
    ic_df = pd.DataFrame(ic_records).set_index("layer")
    ic_disp_cols = ["label_expr", "n_pred", "cs_ic_mean", "cs_ic_std",
                    "cs_ic_ir", "cs_ic_pos_pct", "ts_ic_target"]
    ic_disp_cols = [c for c in ic_disp_cols if c in ic_df.columns]
    ic_html = ic_df[ic_disp_cols].round(4).to_html(classes="tbl", border=0)

    # 信号统计表
    sig_rows = []
    for name, info in layers.items():
        s = info["pred"].dropna()
        sig_rows.append({
            "layer": name,
            "count": len(s),
            "mean": s.mean(), "std": s.std(),
            "min": s.min(), "p5": s.quantile(0.05),
            "p50": s.median(), "p95": s.quantile(0.95), "max": s.max(),
        })
    sig_df = pd.DataFrame(sig_rows).set_index("layer").round(4)
    sig_html = sig_df.to_html(classes="tbl", border=0)

    # 信号直方图
    sig_hist_html = ""
    for name, info in layers.items():
        color = {"outer": "#ef4444", "middle": "#f59e0b", "inner": "#3b82f6"}[name]
        sig_hist_html += _hist_html(info["pred"].dropna(),
                                     f"[{name}] 测试期信号分布",
                                     color=color)

    # IC 时间序列
    ic_ts_html = ""
    for rec in ic_records:
        if "cs_ic_series" in rec and not rec["cs_ic_series"].empty:
            df_ic = pd.DataFrame({"cs_ic": rec["cs_ic_series"]})
            ic_ts_html += _line_html(df_ic, f"[{rec['layer']}] 截面 IC 时间序列")

    # target 三层信号 + close
    target_chart_html = ""
    if target_series and not close.empty:
        merged = pd.DataFrame({k: v for k, v in target_series.items()})
        # 标准化 close 到副坐标
        target_chart_html = _line_html(merged, f"[{target}] 三层信号时间序列")
        # 单独画 close
        close_df = pd.DataFrame({"close": close})
        target_chart_html += _line_html(close_df, f"[{target}] 收盘价")

    # 决策路径
    if not decision_counts.empty:
        decision_html = _bar_html(decision_counts,
                                   f"[{target}] 三层信号 → 决策路径计数",
                                   color="#8b5cf6")
        decision_pct = (decision_counts / decision_counts.sum() * 100).round(2)
        decision_table_html = decision_pct.to_frame("占比 %").to_html(classes="tbl", border=0)
    else:
        decision_html = "<p><i>target 信号缺失，无法计数</i></p>"
        decision_table_html = ""

    # 标签分布
    label_dist_html = "".join(label_dist_blocks.values())
    if label_class_pcts:
        rows = []
        for name, pct in label_class_pcts.items():
            for cls, p in pct.items():
                rows.append({"layer": name, "class": cls, "占比": round(float(p), 4)})
        label_class_html = pd.DataFrame(rows).to_html(
            classes="tbl", index=False, border=0
        )
    else:
        label_class_html = ""

    # 特征重要性
    importance_html = "".join(importance_blocks.values())

    # 互相关
    cross_corr_section = (
        f"<h3>三层信号互相关（{target} 时间序列）</h3>{cross_corr_html}"
    )

    # 总 HTML
    plotly_cdn = (
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
        if HAS_PLOTLY else ""
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>三层 CatBoost+DEnsemble 模型分析报告</title>
{plotly_cdn}
<style>
  body {{ font-family: -apple-system, "Segoe UI", "PingFang SC", sans-serif;
         margin: 24px; color: #1f2937; background: #f9fafb; }}
  h1 {{ color: #111827; border-bottom: 3px solid #3b82f6; padding-bottom: 8px; }}
  h2 {{ color: #1f2937; margin-top: 32px; padding: 6px 12px;
        background: #e0e7ff; border-left: 4px solid #4f46e5; }}
  h3 {{ color: #374151; margin-top: 20px; }}
  .tbl {{ border-collapse: collapse; margin: 8px 0; }}
  .tbl th, .tbl td {{ border: 1px solid #d1d5db; padding: 6px 12px;
                     text-align: right; font-size: 13px; }}
  .tbl th {{ background: #f3f4f6; font-weight: 600; }}
  .tbl tr:nth-child(even) {{ background: #fafafa; }}
  .meta-row {{ display: flex; gap: 20px; flex-wrap: wrap; }}
  .meta-card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
                padding: 12px 16px; min-width: 160px; }}
  .meta-card .v {{ font-size: 18px; font-weight: 600; color: #4f46e5; }}
  .meta-card .k {{ font-size: 12px; color: #6b7280; }}
  pre {{ background: #f3f4f6; padding: 8px; border-radius: 4px;
         font-size: 12px; overflow-x: auto; }}
</style>
</head>
<body>

<h1>📊 三层 CatBoost+DEnsemble 模型分析报告</h1>

<div class="meta-row">
  <div class="meta-card"><div class="k">artifacts</div>
    <div class="v" style="font-size:13px;">{art_dir.name}</div></div>
  <div class="meta-card"><div class="k">target stock</div>
    <div class="v">{target}</div></div>
  <div class="meta-card"><div class="k">报告生成时间</div>
    <div class="v" style="font-size:13px;">{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</div></div>
</div>

<h2>1. 三层模型架构总览</h2>
{overview_html}

<h2>2. 三层 IC 评估</h2>
<p><b>IC 解读</b>：cs_ic_mean = 截面 IC 日均值（理想 &gt; 0.03）；
cs_ic_ir = 信息比率（IC mean / IC std，理想 &gt; 0.3）；
cs_ic_pos_pct = IC 为正的天数占比（理想 &gt; 55%）；
ts_ic_target = 在目标股票上的时序 IC（理想 &gt; 0.05）。</p>
{ic_html}

<h3>截面 IC 时间序列</h3>
{ic_ts_html}

<h2>3. 三层信号统计</h2>
{sig_html}
<h3>信号分布直方图</h3>
{sig_hist_html}

<h2>4. 标的（{target}）三层信号 vs 收盘价</h2>
{target_chart_html}

{cross_corr_section}

<h2>5. 决策路径分布</h2>
<p>使用与 MultiLayerStrategy 一致的阈值 (outer_upper={strategy_outer_upper},
outer_lower={strategy_outer_lower}, middle_buy={strategy_mid_buy},
middle_sell={strategy_mid_sell}, inner_buy={strategy_inner_buy},
inner_sell={strategy_inner_sell})。</p>
{decision_table_html}
{decision_html}

<h2>6. 真实标签分布（{target}）</h2>
{label_class_html}
{label_dist_html}

<h2>7. 子模型特征重要性</h2>
{importance_html}

</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")
    metrics_path = art_dir / "signal_metrics.json"
    metrics = {
        "artifacts_dir": str(art_dir),
        "target": target,
        "sample_pool_for_label": int(sample_pool_for_label),
        "ic": [
            {
                k: v
                for k, v in rec.items()
                if k != "cs_ic_series"
            }
            for rec in ic_records
        ],
        "signal_stats": sig_df.reset_index().to_dict(orient="records"),
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n[OK] 报告已生成: {out_path}")
    print(f"     文件大小: {out_path.stat().st_size / 1024:.1f} KB")
    print(f"[OK] 结构化指标: {metrics_path}")


if __name__ == "__main__":
    fire.Fire({"run": run})
