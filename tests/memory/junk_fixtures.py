"""The memories the dev database actually held on 2026-09-14, verbatim.

Each one was written by the policy as it stood; each one is wrong for a reason
the memory refactor plan names. They are kept as data rather than as prose in a
test so a future rule change is measured against what really happened.

The rows are shaped as :class:`~agentic_erp_assistant.memory.models.
MemoryCandidate` field sets plus the turn context (``request``, ``response``,
``route``) the turn that produced them ended with -- the turn context is what
``SessionMemory.consolidate``'s skip is judged on, while the candidate fields
are what ``policy.decide`` sees.
"""

JUNK_ROWS: tuple[dict, ...] = (
    # (a) written from a turn whose route was "refuse"; also an absence claim
    dict(kind="fact", key="user_name_unavailable",
         statement="The assistant cannot access or retrieve the user's name from the current data sources.",
         request="what is my name ?",
         response="I cannot access personal information such as your name.",
         route="refuse", confidence=1.0),
    dict(kind="fact", key="project_atlas_sprints_unavailable",
         statement="The number of sprints for project Atlas is not available in the current data sources.",
         request="how many sprints project atlas having",
         response="The number of sprints for project Atlas is not available in the current data sources.",
         route="refuse", confidence=0.9),
    dict(kind="fact", key="sprint_information_unavailable",
         statement="No sprint information was retrieved for the project in the current query.",
         request="sprints of project",
         response="Which specific sprint or aspect of the sprints are you interested in?",
         route="clarify", confidence=0.9),
    # (b) restates the assistant's own reply (a document held it); volatile
    dict(kind="fact", key="sprint_13_progress",
         statement="Sprint 13 has completed 22 out of 40 points, with 4 days remaining.",
         request="sprint 13",
         response="Sprint 13 has completed 22 out of 40 points, with 4 days remaining.\n\nSources: sprint-13-report",
         route="answer", confidence=0.9),
    # (c) a preference the user never stated -- inferred from a clarification
    dict(kind="preference", key="clarity_in_requests",
         statement="The user prefers to specify which sprint or aspect of sprints they are interested in when asking about sprints.",
         request="what about sprints ?",
         response="The request is too vague. Please specify which sprint or aspect of sprints you are interested in.",
         route="refuse", confidence=0.8),
)

DRIFTED_PREFERENCE_PAIRS: tuple[tuple[str, str], ...] = (
    # dev DB 2026-09-14 22:39/22:42, keys budget_reporting_format /
    # budget_reporting_currency -- both live at once, both pinned into the
    # next turn's prompt, and the reply used neither: "$187,200".
    ("The user prefers budget numbers to be reported in thousands of USD.",
     "The user prefers budget numbers to be reported in thousands of VND."),
    # dev DB 2026-09-14, keys atlas_reply_prefix / project_atlas_prefix_request
    ("The user wants information about Project Atlas to be prefixed with 'ATLAS 2026'.",
     "The user requested that information about Project Atlas be prefixed with 'ATLAS 2026'."),
)
"""Same preference, two turns, two keys the proposer invented independently --
the drift ``_resolve_conflict``'s exact-key rule cannot see. Each pair is
(the statement already stored, the statement proposed next); see the
fix-memory-key-drift plan, §0."""

DISTINCT_PREFERENCE_PAIRS: tuple[tuple[str, str], ...] = (
    ("The user prefers budget numbers to be reported in thousands of USD.",
     "The user prefers replies written in Vietnamese."),
    ("The user prefers replies written in Vietnamese.",
     "The user wants information about Project Atlas to be prefixed with 'ATLAS 2026'."),
)
"""Two preferences that must never merge -- the negative fixtures for
``TOPIC_OVERLAP_RATIO``, scoring 0.29 and 0.14 against the real ``_overlap``."""

GOOD_ROWS: tuple[dict, ...] = (
    # a real, user-stated preference -- must still be stored after the change
    dict(kind="preference", key="project_atlas_prefix",
         statement="The user wants replies about project Atlas to begin with the prefix 'ATLAS 2026'.",
         request="from now, when i want to see anything about project atlas, you have to begin with 'ATLAS 2026'",
         response="Noted: replies about project Atlas will begin with 'ATLAS 2026'.",
         route="answer", confidence=0.9),
    dict(kind="decision", key="cutover_window",
         statement="The team chose a Thursday evening cutover for the finance module.",
         request="we decided the finance cutover happens thursday evening, note that",
         response="Noted: the finance module cutover is Thursday evening.",
         route="answer", confidence=0.9),
    dict(kind="fact", key="vendor_contact",
         statement="The vendor contact for Atlas is the delivery lead, not procurement.",
         request="fyi the vendor contact for atlas is me, the delivery lead, not procurement",
         response="Noted: the vendor contact for Atlas is the delivery lead.",
         route="answer", confidence=0.9),
)