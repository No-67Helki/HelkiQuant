# Market Data Pipeline

## Source Contract

RQData is the primary source. Local daily CSVs are a historical fallback after
an overlap quality audit. Minute data is always materialized from one provider
because cross-provider adjustment scales are not safe to splice.

Daily front-adjusted prices require an overlapping calibration interval. For
each symbol, the pipeline estimates the fallback-to-primary scale from the
first five common trading sessions, rescales only price fields, and then lets
RQData win on duplicate dates. Volume and turnover are not rescaled.

The canonical readiness audit rejects:

- incomplete daily, PIT, or minute coverage;
- daily data that omits a stock listed at any time inside the holdout;
- PIT-universe dual-source symbols without stable scale evidence;
- missing active Top150 minute symbols;
- minute row counts that differ from the provider-session calendar;
- an untouched window shorter than 60 trading sessions.

## Version Refresh

Use a new version name for every data cutoff:

```powershell
conda activate env310
helki-refresh-canonical `
  --config configs/rqdata_source.local.json `
  --end-date YYYY-MM-DD `
  --target-symbols outputs/rqdata_sync/frozen_top150_symbols.txt
```

The refresh executes these steps in order and stops on the first failure:

1. Fetch all historical common-stock metadata.
2. Derive the union of stocks listed on any holdout session, including stocks
   delisted before the end date.
3. Refetch the daily overlap and PIT ST/suspension state for that dynamic union.
4. Refetch the complete Top150 minute holdout from RQData.
5. Materialize versioned daily and primary-only minute CSVs.
6. Produce a canonical readiness report.

The default daily calibration starts 90 calendar days before the untouched
cutoff. If a long suspension still leaves fewer than five common sessions,
extend `--overlap-start`; never lower the scale-quality threshold to force a
build through.

The full minute window is intentionally refetched. Incremental front-adjusted
minute chunks can acquire incompatible scales after a corporate action.

## Provider Build

After data integrity passes, build the middle provider from the canonical daily
directory. The command uses project-owned provider serialization and the
installed Qlib package only as a research runtime:

```powershell
python -m helki_quant.research.build_pit_daily_pool `
  --mode build `
  --raw-dir data/market_data/canonical/VERSION/daily_qfq `
  --stage-dir data/market_data/staging/VERSION_middle_csv `
  --output-dir data/market_data/providers/VERSION_middle `
  --vwap-mode close
```

Build the outer regime provider from that middle provider without changing the
frozen `broad_adverse_loss5_20d` label definition.

## Evaluation Boundary

Do not inspect strategy returns, change factors, or choose a new Profile before
the canonical report reaches 60 complete post-cutoff sessions. At 60 sessions,
run one frozen untouched replay. Only a passing frozen replay may proceed to a
new target package and at least 20 subsequent audited GmQuant PAPER sessions.

Build promotion evidence only after the canonical report passes. Every frozen
profile must use the exact same audited calendar:

```powershell
helki-promote-frozen build `
  --contract PATH_TO_FROZEN_CONTRACT.json `
  --canonical-readiness outputs/canonical_VERSION_readiness.json `
  --profile-log PROFILE_ID=PATH_TO_PRODUCTION_STYLE_REPLAY `
  --output outputs/frozen_untouched_evidence.json

helki-promote-frozen validate `
  --contract PATH_TO_FROZEN_CONTRACT.json `
  --evidence outputs/frozen_untouched_evidence.json `
  --output outputs/frozen_promotion.json
```

Repeat `--profile-log` and, where required by the frozen contract,
`--profile-manifest` once per profile. Evidence creation is immutable and binds
the exact canonical readiness report, canonical manifest, and session-calendar
hash. A changed source artifact, different 60-session calendar, or legacy
unbound evidence fails closed.

## Current Evidence Snapshot

The `2026-07-31` canonical build has 39 complete post-`2026-06-05`
sessions. Daily coverage includes all 5,215 stocks listed at any point in the
window; PIT state contains 203,385 rows with no missing symbol-date; and the
frozen Top150 minute window passes its source and row-count checks. The frozen
Profile has not been evaluated on this window. Another 21 complete sessions
are required before the one-time untouched replay.

RQSDK license validity must be monitored during collection. A license expiry
is a data-availability failure and must not be bypassed with partial local
updates.
