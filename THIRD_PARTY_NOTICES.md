# Third-Party Notices

HelkiQuant contains original project code and integrates with optional
third-party software. Those components remain subject to their own licenses and
service terms.

## Microsoft Qlib

Research workflows optionally depend on Microsoft Qlib (`pyqlib==0.9.7`). Qlib
is Copyright Microsoft Corporation and is distributed under the MIT License.
The upstream Qlib source tree is not included in this repository.

Project-owned execution code under `helki_quant.research.execution` replaces
the need to redistribute locally modified Qlib backtest files. Any remaining
substantially derived model interfaces must continue to preserve applicable
Microsoft copyright and MIT notices.

## CatBoost

Model training uses CatBoost. CatBoost remains subject to its own license and
notices.

## Ricequant SDK And RQData

RQSDK/RQData are separately installed proprietary services. No SDK package,
license key, or downloaded dataset is redistributed by this repository. Use is
subject to the provider's agreement and data entitlements.

## GmQuant

The deployment adapter targets the GmQuant Python SDK. The SDK, market access,
and simulation services are not redistributed and remain subject to the
platform's terms.
