# Invalid protocol run — do not report

This run was interrupted during training before formal test evaluation. It is
retained only as an audit trail and must not be aggregated, plotted, cited, or
released as an experimental result.

Reason: the provenance implementation allowed Git to search ancestor
directories. Although this machine's unrelated ancestor repository had no
resolvable `HEAD` and the recorded `git` field was therefore `null`, the
implementation posed a structural risk of attributing a future run to the
wrong repository. The implementation and regression tests were corrected
before restarting the complete matrix from scratch.

No `test_metrics.json` or aggregate result exists for this interrupted run.
