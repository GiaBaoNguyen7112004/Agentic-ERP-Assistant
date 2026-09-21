# ui/ — the Agentic ERP Assistant's web client

React 19 + TypeScript + Vite 7, built with Tailwind v4 and hand-written shadcn
components (see `components.json`; primitives under `src/components/ui/`, project
components under `src/components/shared/`). One page, three columns: sidebar
(actor, sessions, pending approvals) · chat · trace, served by FastAPI from
`web/static/` (`npm run build`, git-ignored).

Scripts: `npm run dev` (Vite dev server proxying `/api` to :8000) ·
`npm test` (vitest, jsdom) · `npm run typecheck` · `npm run lint` (oxlint) ·
`npm run build`.

Structure:

- `src/protocol.ts` mirrors `web/protocol.py` field for field —
  `tests/web/test_protocol_drift.py` fails the Python suite when they drift.
- `src/turnReducer.ts` is the pure, tested core: one function turns the SSE
  stream into a `TurnView`; every rendered turn reads one.
- `src/{appState,useTurnStream,sse,api}.ts`: state, the stream hook, the SSE
  parser, fetch wrappers — none of it changes with presentation.
- `src/lib/{routeBadge,traceTone,format}.ts` are the pure presentation maps;
  badge labels are quoted verbatim in `docs/manual-test.md` §3.
- Dark mode follows the OS; tokens live in `src/index.css`.