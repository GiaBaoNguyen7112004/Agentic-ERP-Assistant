1. role: you are the solution architect engineer for this ERP AGENTIC assistant, investigate this project to make sure you know all about the architecture, pattern, technical decisions and missing piece and acceptable gaps of this project.

2. problem:

- current we have pieces in the core, but we dont wire it together to prove it working properly.
- we dont have any interface or web UI so we can not test all of them.

3. goals:

- investigate all the tools scenarios, RAG scenarios, memory scenarios, graph scenarios to know the insight business and all the scenario we have to test to prove our system working correctly
- build the plan to wiring everything together so user can test it via the WEB
- build a web chat with the assistant, for now dont it a complicated login feature, just have a switch between users feature directly on the web (for dev to test, dont need to login). the chat should be streaming instead wait all the answer so that user can see the streaming text. beside the chat we the assistant send to user, we have to send the agent decisions, citations, ...etc ( trace ).
- build the handbook with all the scenario i will test and each case you will provide the example request, what flow we expect, how to verify that data in database or the UI.

4. output

- a detail e2e-code-plan.md contain details step so that sonnent 5 can follow it to implement
- a detail manual-test.md contain details test scenarios.
- open any important missing piece need to implement to finish the first version of this end to end.
