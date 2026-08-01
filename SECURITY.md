# Security

## Credentials

Never commit API licenses, platform tokens, account identifiers, target files,
position snapshots, or order audits. Use environment variables and local
Git-ignored secret files.

The public repository must not contain values for `GM_TOKEN`, `GM_ACCOUNT_ID`,
or an RQSDK license. Run the publication scan before every release that changes
deployment configuration.

## Trading Safety

The checked-in GmQuant wrapper is intended for simulation/PAPER accounts. A
caller must explicitly provide the account and fresh generated target files.
Real-money connectivity, permissions, and account-specific controls are outside
the repository's default configuration.

The intraday overlay remains no-order by default. Removing that guard requires
separate portfolio replay, order audit, and PAPER acceptance evidence.

## Data Handling

Market datasets, model binaries, predictions, account captures, and generated
audits are excluded from Git. They may contain licensed or sensitive material
and should be stored in access-controlled local or object storage.

## Reporting

For a security issue, open a private security advisory in the GitHub repository
with reproduction steps and affected versions. Do not include credentials,
account identifiers, or proprietary data samples in a public issue.
