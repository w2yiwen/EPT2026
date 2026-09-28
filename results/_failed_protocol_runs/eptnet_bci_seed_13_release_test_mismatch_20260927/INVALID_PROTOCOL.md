# Invalid protocol run — do not report

This run was interrupted during training before formal test evaluation. It is
retained only as an audit trail and must not be aggregated, plotted, cited, or
released as an experimental result.

Reason: the release allowlist correctly removed the superseded continuous
report builder, but a provenance-tracked newline regression test still tried
to open that removed builder. A primary-only staged copy would therefore fail
its own full test suite. The stale test target and all other public-source
references were audited before freezing a new source identity and restarting
the complete matrix from scratch.

No `test_metrics.json` or aggregate result exists for this interrupted run.
