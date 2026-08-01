# Architecture

## Boundaries

HelkiQuant separates research from execution.

- `helki_quant.research` owns data normalization, factor and label generation,
  model training, walk-forward OOF assembly, portfolio replay, stress testing,
  and promotion gates.
- `helki_quant.deployment` owns the compact GmQuant runtime. It consumes frozen
  target and risk artifacts and does not import Qlib.
- Microsoft Qlib is an optional installed research dependency. Its upstream
  source tree is not part of this repository.

## Data Flow

1. The isolated RQData bridge downloads adjusted bars, calendars, instrument
   metadata, and PIT ST/suspension state.
2. The source gateway merges API rows with local fallback rows. API data wins
   on equal timestamps.
3. Quality reports must pass before canonical materialization.
4. Daily and minute providers are built into versioned local directories that
   remain outside Git.
5. Purged walk-forward training produces fold-isolated predictions.
6. Portfolio replay converts OOF predictions into cost-aware target portfolios.
7. Frozen promotion gates select a PAPER candidate without reusing untouched
   evaluation data for parameter selection.

## Strategy Contract

The middle layer is the stock-selection authority. Membership is recomputed
only on scheduled provider sessions. Between scheduled rebalances, target
shares are carried unchanged except for mandatory exits such as refreshed ST
or delisting exclusions.

The outer layer may reduce portfolio exposure under adverse regimes. It does
not select stocks and is disabled unless strict OOF comparisons improve risk.

The inner layer sees only actual held positions and only information available
at each intraday decision time. It cannot enter the order path until its own
portfolio replay and PAPER evidence gates pass.

## Execution Safety

- Required simulation-account binding through environment configuration.
- Dynamic ST, delisting-name, suspension, and limit-state checks.
- Existing-position synchronization before target comparison.
- A-share T+1 availability and sell reservation tracking.
- Cash, whole-lot, minimum-order, turnover, and concentration constraints.
- Target freshness and signal-to-target date validation.
- Atomic audit artifacts and hash-chained PAPER activation records.
- Fail-closed behavior for missing metadata, stale targets, duplicate launches,
  unresolved sell risk, or unexpected order rejection.

## Dependency Boundary

The GmQuant runtime uses project-owned execution logic and has no Qlib import.
Research modules may use the installed `pyqlib==0.9.7` interfaces for provider,
dataset, processor, and model integration. This keeps third-party framework
code replaceable and avoids shadowing the installed package with vendored
source.
