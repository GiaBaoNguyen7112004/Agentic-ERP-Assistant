# Sprint 13 Review and Retrospective — Atlas

Sprint 13 runs 2026-08-24 to 2026-09-10. This note is written at the mid-sprint
checkpoint on 2026-09-06 and covers the review of committed work, the burn-down
position, and the retrospective actions carried forward from Sprint 12.

## 1. Commitment and progress

The team committed 40 story points. As of 2026-09-06, 22 points are complete and
accepted, with 4 working days remaining in the sprint. The required run rate to
land the commitment is 4.5 points per day against an observed sprint-to-date
rate of 3.1.

### 1.1 What is complete

Twenty-two points across six stories, all in the finance workstream: the cost
centre mapping rules (8 points), the reconciliation exception report (5), the
journal import validation pass (5), and three defect fixes (2, 1, 1).

### 1.2 What is at risk

The remaining 18 points sit in two stories. The larger, the payments interface
regression harness at 13 points, is blocked on the vendor test window described
in the September status report, section 3.2. It will not complete this sprint.
The smaller, at 5 points, is expected to land.

The team forecast is 27 of 40 points, a 68% commitment hit rate. Sprint 12
delivered 34 of 34.

## 2. Why the commitment was missed

The 13-point story was committed on the assumption that the vendor test window
would be confirmed during the first week of the sprint. It was not. The team
knew the dependency was unconfirmed at planning and committed the story anyway,
because the alternative was to leave capacity unallocated.

This is the second sprint in which an unconfirmed external dependency has been
pulled into a commitment. Sprint 11 lost 8 points the same way.

## 3. Retrospective

### 3.1 Actions from Sprint 12

Two actions were carried into this sprint. The first, to add a pre-planning
dependency check, was completed on 2026-08-26 and is now part of the planning
agenda. The second, to reduce the review queue by pairing on acceptance, was not
done; the reviewer bottleneck is unchanged and is carried forward again.

### 3.2 New actions

1. Do not commit a story whose external dependency is unconfirmed at planning.
   Place it in the sprint backlog as a stretch item instead. Owner: scrum master,
   effective Sprint 14.
2. Raise the vendor test window as a formal contractual escalation rather than a
   fourth written request. Owner: Priya Raman, by 2026-09-09.
3. Timebox the reconciliation exception triage to two hours per day. The team
   spent an estimated 14 hours on ad-hoc triage this sprint against a planned 8.

## 4. Velocity

Rolling three-sprint velocity is 31 points: Sprint 11 delivered 26 of 34,
Sprint 12 delivered 34 of 34, and Sprint 13 is forecast at 27 of 40. The team
has asked that Sprint 14 be planned at 30 points rather than 40 until the
external dependency practice in section 3.2 has held for a full sprint.
