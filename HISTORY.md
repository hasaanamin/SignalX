# Retrospectively imported development history

Development predated Git tracking. Historical author dates below were supplied
by the maintainer as approximate milestone dates. They are not claims that Git
commits existed on those days. Committer dates record the actual import time.
Archive file timestamps may differ from the chosen author dates.

Each historical commit imports real saved source, rather than dividing today's
final implementation into fabricated intermediate revisions. Historical trees
are not asserted to pass current tests or to be production-ready.

| Approximate author date | Imported milestone | Source |
|---|---|---|
| 2026-09-18 | Original DSP/feature prototypes | `packages.zip` |
| 2026-09-20 | Modular detection/capture foundation | `pyproject_baseline.zip` |
| 2026-09-25 | Initial mechanism repairs | `pyproject.zip` |
| 2026-09-27 | Examples and detector iteration | `examples.zip` |
| 2026-09-30 | 2.1 reliability repair snapshot | `examples2_repaired.zip` |
| 2026-10-03 | 2.2 lifecycle/policy repairs | `examples2_repaired_2.2.0.zip` |
| 2026-10-04 | 2.3 numerical/lifecycle repairs | `examples2_repaired_2.3.0.zip` |
| 2026-10-09 | Acquisition runtime update | Verified acquisition-stage2 2.4.0rc1 wheel modules |
| 2026-10-10 | Current dataset UI and AR1 candidate | Active source snapshot at import |

The acquisition runtime commit imports the verified wheel's production modules
over the preceding tree; it is not a claim that the preceding tests were that
wheel's original complete checkout. The final active-source import restores
the actual current code/test/benchmark layout. A separate current-dated commit
adds repository housekeeping, this documentation and Git LFS dataset tracking.

Original archives and audit evidence are preserved in the maintainer's local
recovery archive, not duplicated into the current checkout. Imported commit
messages record snapshot provenance and SHA-256 where applicable. Binary
datasets enter the history through Git LFS in the current-dated commit.
