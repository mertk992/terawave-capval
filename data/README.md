# Synthetic Enterprise Data

`synthetic_enterprise_data.json` is the source corpus used by the TeraWave demo.
Every record is fabricated for the final project and does not represent actual
Blue Origin data.

The file is intentionally shaped like enterprise source material:

- `budget_pools`: synthetic ERP budget records
- `historical_requests`: synthetic procurement/workflow precedents
- `approval_tiers`: synthetic corporate finance policy thresholds
- `documents`: synthetic SharePoint-style contracts, policies, memos, and reports

Agent tools include `source_file`, `dataset_id`, and source record IDs in their
outputs so the generated memo can cite this synthetic evidence pack.
