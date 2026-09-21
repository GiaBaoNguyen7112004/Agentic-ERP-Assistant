# Atlas Post-Go-Live Support and Escalation Policy

Version 2.1, effective 2026-07-01. Owner: service management. This policy applies
to every module of the Atlas ERP platform from the moment that module completes
cutover, and it supersedes version 2.0 in full.

## 1. Scope and definitions

### 1.1 What this policy covers

Incident handling, escalation, and the service levels the delivery organisation
commits to during hypercare and steady state. It does not cover change requests,
which follow the change control procedure, or data correction requests, which
follow the data governance procedure.

### 1.2 Hypercare and steady state

Hypercare is the six weeks following a module cutover. During hypercare the
delivery team retains first-line responsibility and response targets are halved.
Steady state begins at the end of the sixth week, at which point first line
transfers to the service desk.

## 2. Incident severity

### 2.1 Severity definitions

**Severity 1.** The module is unavailable, or a financial period cannot be
closed. Any incident that prevents payment of suppliers or staff is severity 1
regardless of how many users are affected.

**Severity 2.** A core process is degraded and no workaround exists, or a
workaround exists but cannot be sustained for more than one working day.

**Severity 3.** A core process is degraded and a sustainable workaround exists,
or a non-core process is unavailable.

**Severity 4.** Cosmetic defects, documentation errors, and enhancement
suggestions raised through the incident channel.

### 2.2 Who sets severity

The service desk sets an initial severity on logging. The incident manager may
change it at any point. A requester may dispute a severity once; the dispute is
decided by the incident manager and is not escalated further.

### 2.3 Response and resolution targets

| Severity | Response, hypercare | Response, steady state | Resolution target |
|---|---|---|---|
| 1 | 15 minutes | 30 minutes | 4 hours |
| 2 | 30 minutes | 1 hour | 1 working day |
| 3 | 4 hours | 1 working day | 5 working days |
| 4 | 1 working day | 5 working days | Next release |

Response means a named engineer has acknowledged the incident and begun work.
It does not mean an automated ticket confirmation.

## 3. Escalation within the delivery organisation

### 3.1 Automatic escalation

A severity 1 incident escalates to the delivery lead automatically at the
response target and to the programme sponsor at twice the resolution target. No
one needs to request this; the incident tool raises it.

### 3.2 Requested escalation

Any user may request escalation through the service desk. A requested escalation
that is refused must be recorded with a reason, and the record is reviewed at
the monthly service review.

## 4. Escalation to a vendor

### 4.1 When contractual escalation applies

Where an incident or a delivery dependency sits with a third-party vendor and
the vendor has not responded within the period set out in the relevant master
services agreement, the matter moves to contractual escalation. Three unanswered
written requests are sufficient grounds; a fourth request is not required and
should not be sent.

### 4.2 How contractual escalation is raised

Contractual escalation is raised in writing to the account director of the
vendor by the workstream owner, copied to procurement and to the programme
sponsor. The letter must cite the specific clause relied on, state the response
deadline, and state whether service credits are being claimed. The standard
response deadline is five working days.

### 4.3 Service credits

Service credits are claimed by procurement, never by the delivery team, and are
claimed against the quarter in which the breach occurred. A claim that is not
raised within 60 days of the breach is time barred under the standard terms.

## 5. Reporting

A service report is published monthly, covering incident volumes by severity,
target attainment, the top five recurring causes, and every escalation raised in
the period. The report is circulated to the steering committee and is not
confidential.
