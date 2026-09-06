# Atlas ERP Rollout — Delivery Status Report, September 2026

Reporting period: 2026-08-01 to 2026-08-31. Prepared by the delivery lead for the
Atlas steering committee and the programme management office. Figures in this
report are taken from the delivery tracker on 2026-08-31 and reconciled against
the finance ledger extract of the same date.

## 1. Summary

Atlas remains **amber**. The warehouse workstream is on plan and the finance
workstream is not. Milestone M2, the finance module cutover, is two days behind
its 2026-09-11 due date and is the only milestone currently carrying schedule
risk. Nothing in this period has changed the overall go-live position for M3.

The single decision the steering committee is asked to take this month is
whether to accept a two-day slip on M2 or to fund a second migration rehearsal
weekend to recover it. Section 5 sets out both options with their costs.

### 1.1 Status by workstream

| Workstream | RAG | Movement since July | Owner |
|---|---|---|---|
| Finance | Amber | Was green; downgraded 2026-08-19 | Priya Raman |
| Warehouse | Green | Unchanged | Tomas Lindqvist |
| Integrations | Amber | Unchanged | Priya Raman |
| Data migration | Amber | Was amber; unchanged | Wei Chen |
| Change and training | Green | Was amber; upgraded 2026-08-25 | Anna Berg |

### 1.2 What changed this period

The finance workstream was downgraded to amber on 2026-08-19 after the second
migration rehearsal produced 41 unresolved reconciliation exceptions against a
tolerance of 10. Change and training was upgraded to green on 2026-08-25 when
the last of the four regional training cohorts completed with a 94% attendance
rate.

## 2. Milestones

### 2.1 M1 — Discovery and scope sign-off

Completed on 2026-05-29, on its due date. The signed scope baseline is the
version of record for all change control since; three change requests have been
raised against it, two approved and one rejected.

### 2.2 M2 — Finance module cutover

Due 2026-09-11, currently forecast at 2026-09-13 — two days late. The slip is
entirely attributable to the reconciliation exceptions described in section 3.1
and does not reflect any scope growth. The cutover window itself is a 52-hour
freeze beginning at 18:00 on the Friday; a two-day slip moves the window into
the following weekend rather than extending it.

### 2.3 M3 — Warehouse module cutover

Due 2026-11-06, on track. Warehouse has completed integration testing for all
seven inbound interfaces and is holding four weeks of float. The steering
committee has previously agreed that M3 float may not be consumed to recover M2.

## 3. Delivery risk

### 3.1 Finance data migration

The second migration rehearsal on 2026-08-15 produced 41 reconciliation
exceptions against a tolerance of 10. Thirty-one of the 41 trace to a single
root cause: historic journal entries carrying a cost centre that was retired in
the 2024 restructure and never mapped in the migration ruleset. A mapping fix
has been written and is awaiting a third rehearsal.

This is tracked as risk R-1, severity high, owned by Wei Chen. It is the
principal reason M2 is forecast late.

### 3.2 Second integration vendor

The second integration vendor has not confirmed a test window for the payments
interface, despite three written requests since 2026-07-22. Without a confirmed
window the interface cannot be regression tested before the M2 freeze, and the
fallback is to cut over with the legacy payments bridge in place for the first
two weeks after go-live.

This is tracked as risk R-2, severity medium, owned by Priya Raman. The
contractual escalation path is set out in the support policy, section 4.

### 3.3 Risks closed this period

Two risks were closed: R-5, concerning the availability of the warehouse test
environment, closed on 2026-08-08 after the environment was rebuilt; and R-6,
concerning training room availability, closed on 2026-08-25.

## 4. Budget position

As of 2026-08-31 the programme has spent 292,800 USD of an approved 480,000 USD,
which is 61% of budget against 68% of elapsed schedule. The forecast at
completion is 515,000 USD, an overrun of 35,000 USD against approval.

The overrun is driven by the two additional migration rehearsals and by extended
vendor support hours in August. The detailed breakdown, including the variance
analysis by cost category, is in the Q3 budget summary and is restricted to the
finance group.

## 5. Decision requested

### 5.1 Option A — accept the two-day slip

Move the M2 cutover window to 2026-09-18. Costs nothing directly, consumes no
M3 float, and delays the first finance close on the new system by one period.
The finance director has indicated this is acceptable if it is decided before
2026-09-10.

### 5.2 Option B — fund a recovery rehearsal

Run a third migration rehearsal on the weekend of 2026-09-05 with vendor support
on standby. Estimated cost 18,000 USD, of which 11,000 USD is vendor support at
weekend rates. Recovers the original 2026-09-11 date if the rehearsal clears; if
it does not, the slip is unchanged and the 18,000 USD is spent regardless.

The delivery lead recommends Option A. The reconciliation fix has not yet been
proven in any rehearsal, so Option B buys a chance rather than a date.

## 6. Next period

Third migration rehearsal, M2 cutover readiness review on 2026-09-09, and the
first of the two hypercare planning workshops.
