# Label and target-speaker direction audit

Status: **passed**

Frozen semantics: `0=deception`, `1=truth`; metrics use class 0 as positive.

| Session | Split | Target speaker | Valid steps | Deception | Prevalence | Direction mismatches |
|---|---|---:|---:|---:|---:|---:|
| session_002 | train | 1 | 223 | 46 | 0.206 | 0 |
| session_003 | train | 2 | 340 | 29 | 0.085 | 0 |
| session_004 | train | 2 | 150 | 26 | 0.173 | 0 |
| session_006 | train | 2 | 106 | 58 | 0.547 | 0 |
| session_009 | train | 2 | 133 | 10 | 0.075 | 0 |
| session_010 | train | 1 | 324 | 57 | 0.176 | 0 |
| session_012 | train | 2 | 352 | 113 | 0.321 | 0 |
| session_018 | val | 1 | 260 | 63 | 0.242 | 0 |
| session_008 | test | 2 | 186 | 56 | 0.301 | 0 |
| session_015 | test | 2 | 147 | 12 | 0.082 | 0 |
| session_017 | test | 1 | 312 | 71 | 0.228 | 0 |

## Limitation

Passing proves internal direction and speaker-provenance consistency; it cannot prove that the original human highlights are semantically correct. Independent data-owner confirmation remains required.
