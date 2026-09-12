# UI Redesign Plan — `ui/` (Agentic ERP Assistant)

Scope check first: the frontend is **React 19 + TypeScript + Vite 7** (not Next.js) — one
page, no router, three columns, plain CSS in `ui/src/index.css`, no component library. The
plan below is adapted to that (no `next-themes`, no `app/` directory).

This is a **UI/UX-only** plan. Business logic, API contracts, data fetching, state
management, routing, permissions, and existing functionality are out of bounds — see §7.

---

## 0. Current state and problems

**Inventory.** `App.tsx` (shell + all handlers) → `ActorSwitcher`, `SessionList`,
`ApprovalQueue` (sidebar) · `ChatPanel` → `MessageList` → `AssistantMessage` →
`CitationChips`, `FailureBlock`, `ApprovalCard`; `Composer` · `TracePanel` → `TurnTrace` →
`DecisionList`, `EventRow`, `TurnSummary`. Logic lives in `appState.ts`, `turnReducer.ts`,
`useTurnStream.ts`, `sse.ts`, `api.ts`, `protocol.ts` — **none of these change.**

**Problems found (all presentation):**

1. **No token system.** Ten arbitrary font sizes (0.7–1.1rem), pixel paddings picked per
   rule, radii 6/8/10/999px, no monospace font although IDs, scopes, tool names,
   locators, and JSON are shown everywhere.
2. **Inline styles** in `App.tsx` (`padding:16`), `ActorSwitcher` (`marginTop`),
   `SessionList` (active = `fontWeight`), `AssistantMessage`/`DecisionList`
   (`whiteSpace`).
3. **Leaky global selectors** (`h2`, `button`, `select, textarea, input`) style everything;
   the uppercase-dim `h2` rule is really a "section label" pattern.
4. **Buttons are four different things:** unstyled native (New chat, queue Approve/Deny,
   Send/Stop) vs `.approval-actions` styled; no disabled/focus-visible/hover states;
   disabled buttons keep `cursor:pointer`.
5. **Session rows are `<li onClick>`** — not focusable, no hover/active affordance,
   `last_started_at` never shown, long `first_request` overflows.
6. **Approval queue** shows `arguments_summary` unlabelled, hides `created_at`/`session_id`,
   duplicates the Approve/Deny pair from `ApprovalCard` with different styling; "Nothing
   pending" is a list item.
7. **Route badge** is plain uppercase grey text — `failed`/`refused`/`waiting for approval`
   carry no colour or icon; streaming shows a bare `…` with no activity indicator; no
   auto-scroll to the newest message.
8. **Trace rows** are four unaligned `<span>`s in a flex row — ragged columns; gateway rows
   are only italic; `kindClass()` colours are hard-coded (`#9333ea` has no dark variant).
9. **Turn summary** is three prose lines of `key: value · key: value`.
10. **Dark mode** flips `--bg/--text/--accent` only; `--danger/--ok/--warn` stay light-mode
    values (low contrast on `#17181c`); user bubble hard-codes `#fff`.
11. **Empty/loading states:** `Loading…` paragraph; empty chat column is blank; trace panel
    says "Ask something…"; a failed `/api/users` leaves "Loading…" forever.
12. **Responsive** (<1000px) stacks sidebar → chat → trace vertically; the composer is
    off-screen below the sidebar.
13. **Rendering bug:** `ApprovalCard` renders `String(value)` → `[object Object]` for any
    non-scalar tool argument.
14. Template leftovers: `ui/README.md` (Vite boilerplate), `public/icons.svg` (unused
    social sprite), Vite favicon.

---

## 1. Design system

Base: **shadcn/ui (Tailwind v4, CSS-variable theme, base colour `neutral`) + Lucide**.
Everything below is expressed as tokens in `ui/src/index.css` and consumed only through
Tailwind utilities / shadcn variants.

| Area | Decision |
|---|---|
| **Colour tokens** | shadcn set (`background, foreground, card, popover, primary, secondary, muted, accent, destructive, border, input, ring, sidebar-*`) **plus four project tones** exposed via `@theme inline`: `--color-success` (tool ok / approve / "can approve"), `--color-warning` (approval pending, refused), `--color-info` (decisions / documents), `--color-memory` (memory events). Each defined for light and dark; verify ≥ 4.5:1 as text on `background` in both. |
| **Dark mode** | Keep today's behaviour: **follows the OS, no toggle.** Put the dark token block under `@media (prefers-color-scheme: dark) { :root { … } }` instead of `.dark {}`, delete the `@custom-variant dark` line `shadcn init` writes (Tailwind v4's default `dark:` is already the media query). Keep `color-scheme: light dark` on `:root`. |
| **Typography** | `--font-sans`: `ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif`; `--font-mono`: `ui-monospace, SFMono-Regular, Menlo, Consolas, monospace` (no webfont dependency). App body `text-sm` (14px). Scale in use: `text-xs` metadata/trace/chips · `text-sm` body, lists, forms · `text-base` message text · `text-lg font-semibold` brand. **Section label** = `text-[11px] font-medium uppercase tracking-wider text-muted-foreground`. **Mono** for every id, scope, tool name, locator, event kind, JSON, cost/token numbers (`tabular-nums`). |
| **Spacing** | 4px grid. Panel padding `p-4`; section gap `space-y-5`; list row `px-2 py-1.5`; card `p-3`; chip `px-1.5 py-0.5`; message gap `space-y-4`. |
| **Radius** | `--radius: 0.5rem`. Buttons/inputs/cards `rounded-md`/`rounded-lg`; badges `rounded-md`; user bubble `rounded-2xl rounded-br-md`. No pills — chips are `rounded-md` (technical product). |
| **Borders** | 1px `border-border` everywhere; sidebar/trace `bg-sidebar` + `border-r`/`border-l`; `Separator` between sidebar sections. Tone borders via `border-warning/50`, `border-destructive/50`. |
| **Buttons** | shadcn `Button` only. `default` = Send, Approve (with `Check`) · `outline` = Deny (`X`, `text-destructive`), Stop (`Square`), New chat (`Plus`) · `ghost` + `size="icon"` = copy id, open drawer, external link. `size="sm"` in sidebar/cards, `default` in composer. Disabled = shadcn defaults (`opacity-50 pointer-events-none`). Icons `size-4`, `aria-hidden` (Lucide default). |
| **Forms** | shadcn `Select` (actor), `Textarea` (composer, inside a bordered `focus-within:ring-2 ring-ring/40` container). Keep `aria-label="Actor"` / `aria-label="Message"`. |
| **Cards** | shadcn `Card` for approval card and queue items; argument lists as `<dl class="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1">`. |
| **Tables** | shadcn `Table` for trace events; dense variant: `text-xs`, `[&_td]:py-1 [&_th]:py-1.5`, `hover:bg-muted/50`. |
| **Dialogs / drawers** | shadcn `Sheet` for sidebar and trace panel below `lg`. No modal dialogs required by current features. |
| **Navigation** | Left sidebar = brand · actor · sessions · approvals. Chat header = context strip (session, actor, running indicator). Trace header = trace id + link. |
| **States** | Loading: `Loader2 animate-spin` or `Skeleton`. Empty: `EmptyState` (icon + one line + hint). Streaming: blinking caret (`animate-pulse` block) after text. Paused: `border-warning`. Failed/error: `Alert variant="destructive"`. Refused: `Badge` warning tone. Focus: `focus-visible:ring-2 ring-ring`. Active session: `bg-accent text-accent-foreground` + `aria-current`. Deciding: buttons disabled + spinner. |

**Semantic maps (single source, data-driven — no `if` chains in JSX):**

| Route badge label (unchanged text) | Tone | Icon |
|---|---|---|
| `documents` | info | `FileText` |
| `tool` | info | `Wrench` |
| `clarify` | neutral | `MessageCircleQuestion` |
| `refused` | warning | `ShieldBan` |
| `failed` | destructive | `CircleAlert` |
| `waiting for approval` | warning | `Hourglass` |
| `starting` / `running` | neutral | `Loader2` spinning |
| anything else (`route ?? 'unknown'`) | neutral | — |

| Trace kind (same substring rules as `kindClass`) | Tone |
|---|---|
| `route_selected` | info |
| `tool_called` | success |
| `approval*` | warning |
| `memory*` | memory |
| `*failed*`, `run_failed` | destructive |

---

## 2. Shared components

**shadcn (generated into `ui/src/components/ui/`):** `button`, `badge`, `card`, `select`,
`textarea`, `separator`, `tooltip`, `table`, `alert`, `skeleton`, `sheet`, `collapsible`.
If the registry has `kbd`, `empty`, `spinner`, use them; otherwise write the tiny
equivalents below.

**Project components (`ui/src/components/shared/`):**

| Component | Props | Replaces |
|---|---|---|
| `SectionHeading` | `{ children, count?: number, action?: ReactNode }` | global `h2` rule |
| `RouteBadge` | `{ turn: TurnView }` → uses `lib/routeBadge.ts` | inline `routeBadge()` + `.route-badge` |
| `ToneBadge` | `Badge` extended with `tone: 'neutral'\|'info'\|'success'\|'warning'\|'destructive'\|'memory'` | ad-hoc colours |
| `MonoId` | `{ value, href?, copy?: boolean, truncate?: boolean }` | raw ids / trace heading |
| `KeyValueList` | `{ items: {key, value: ReactNode}[] }` | `ApprovalCard` `<dl>`, `TurnSummary` |
| `JsonBlock` | `{ value: unknown }` | `DecisionList` `<pre>` |
| `EmptyState` | `{ icon, title, hint? }` | "Nothing pending", "No sessions yet", trace placeholder, blank chat |
| `ApprovalActions` | `{ canApprove, deciding, onDecide, size? }` | duplicated Approve/Deny in `ApprovalCard` + `ApprovalQueue` (button text stays exactly `Approve` / `Deny`) |
| `ScopeChip` | `{ scope }` mono outline badge | `.scope-chip` |
| `StreamingCaret` | none | the `…` placeholder |

**Libs (`ui/src/lib/`):** `utils.ts` (`cn`, from shadcn) · `routeBadge.ts` (pure:
`TurnView → { label, tone, icon }`, labels byte-identical to today's) · `traceTone.ts`
(pure: `kind → tone`) · `format.ts` (`formatRelativeTime(iso, now)`,
`formatCost(number|null)`, `formatArgument(unknown)` = JSON for objects, `String` for
scalars).

---

## 3. Redesign per region (one page, four regions)

**App shell (`App.tsx` JSX only + new `components/layout/AppShell.tsx`).**
`h-dvh grid lg:grid-cols-[280px_minmax(0,1fr)_360px]`; each column `min-h-0 overflow-y-auto`,
composer pinned (keeps the ADR 0015 fix). Below `lg`: chat only, sidebar and trace open as
`Sheet`s from a top bar (`PanelLeft`, `Activity` icons). Loading gate → centred spinner
"Loading actors…" inside the shell. All hooks/handlers in `App.tsx` stay verbatim.

**Sidebar.** Brand row (`Bot` icon, title, `dev` outline badge). `ActorSwitcher`: `Select`
showing `display_name` with `role` as muted second line; below: `project_code` badge,
`ShieldCheck` success badge "can approve", scopes as `ScopeChip`s. `SessionList`:
`SectionHeading "Sessions"` with `New chat` action; rows are `<button>` (`truncate` +
`title`, relative `last_started_at`, `aria-current` on active); `EmptyState`.
`ApprovalQueue`: heading with count badge; each item a `Card`: `tool_name` mono,
"requested by `actor`", `arguments_summary` mono truncated with tooltip, relative
`created_at`, `ApprovalActions size="sm"`, muted "switch to an approver" line when
`!canApprove`.

**Chat column.** New header strip: session id (`MonoId`) or "New chat", actor, `running`
pulse. `MessageList`: native `overflow-y-auto min-h-0`, bottom sentinel + `scrollIntoView`
on new content; user message `bg-primary text-primary-foreground max-w-[80%] self-end`;
assistant message = left block with `Bot` avatar column, meta row (`RouteBadge`, tool
names from `summary.observations` when present), `whitespace-pre-wrap` text +
`StreamingCaret` while `!answer`. `CitationChips`: document chips as `<a>` badges (info
tone, `ExternalLink` icon, tooltip `source_id · locator`); ERP chips neutral with
`Database` icon. `FailureBlock` → `Alert variant="destructive"` (`failure` mono +
`error_detail`). `ApprovalCard` → `Card border-warning/60` with `Hourglass` header,
`KeyValueList` of arguments via `formatArgument`, `ApprovalActions`, keep the
`switch to an approver to decide this` text. Empty chat → `EmptyState` with the existing
placeholder sentence. `Composer`: bordered container, borderless `Textarea`
(`min-h-[2.5rem]`, `resize-y`), footer row with `Kbd` hint "Enter ↵ send · Shift+Enter
newline" and Send (`SendHorizontal`) / Stop (`Square`, outline). Key handling identical.

**Trace panel.** Header: "Trace" + `RouteBadge` + `MonoId` of `traceId` (full id,
`break-all`, copy button, `ExternalLink` to `/api/runs/{id}` — same URL). Decisions:
`SectionHeading` with count, ordered list — `ToneBadge route` `→` `tool_name` mono, args in
`JsonBlock` inside `Collapsible` **default open**. Events: dense `Table` — `#` (`w-8
tabular-nums text-muted-foreground`, `·` for null) · node · kind (mono, `traceTone`) ·
detail (`break-words`); gateway rows `italic text-muted-foreground` with
`CornerDownRight` prefix and `data-source="tool_gateway"`, `title="live only — tool
gateway hook, not persisted"`. Summary: `KeyValueList` in a 2-col grid — outcome (badge),
route, steps · evidence, memories recalled, history shown · model calls, tokens in/out,
cost (`formatCost`), unpriced · observations as mono lines. History-loaded turns
(`events.length === 0 && status !== 'starting'`) show an `EmptyState` "No live trace for
a turn loaded from history" with the run link.

---

## 4. Phases

### Phase 0 — Baseline

- **Objective:** know what green looks like before touching anything.
- **Files:** none.
- **Changes:** none.
- **Verify:** `npm --prefix ui run typecheck && npm --prefix ui test && npm --prefix ui run build && npm --prefix ui run lint`
  all green; note the test count (4 files). Record
  `uv run pytest -q tests/web/test_protocol_drift.py` green. Screenshot the current UI at
  1440 / 1000 / 400 px if a server is available.

### Phase 1 — Tooling: Tailwind v4 + shadcn + alias

- **Objective:** the design-system substrate, app still renders exactly as before.
- **Files:** `ui/package.json`, `ui/vite.config.ts`, `ui/tsconfig.json`,
  `ui/tsconfig.app.json`, `ui/components.json` (new), `ui/src/lib/utils.ts` (new),
  `ui/src/index.css`, `ui/index.html`.
- **Changes:**
  1. `npm --prefix ui install tailwindcss @tailwindcss/vite`.
  2. `vite.config.ts`: `import path from 'node:path'`,
     `import tailwindcss from '@tailwindcss/vite'`; `plugins: [react(), tailwindcss()]`;
     `resolve: { alias: { '@': path.resolve(__dirname, './src') } }` (vitest picks the
     alias up from the same file). Leave `build.outDir`, `emptyOutDir`, `server.proxy`,
     `test` untouched.
  3. Add `"baseUrl": "."` and `"paths": { "@/*": ["./src/*"] }` to `compilerOptions` in
     **both** `tsconfig.json` and `tsconfig.app.json`.
  4. `npx shadcn@latest init -y -b neutral` (run inside `ui/`; if the CLI cannot run
     non-interactively, hand-write `components.json`, `lib/utils.ts`, and the theme block
     from the shadcn Vite docs). Expect `class-variance-authority`, `clsx`,
     `tailwind-merge`, `lucide-react`, `tw-animate-css` to be added.
  5. `index.css` becomes: `@import "tailwindcss"; @import "tw-animate-css";` → `:root`
     light tokens (+ the four project tones, `color-scheme: light dark`) →
     `@media (prefers-color-scheme: dark) { :root {…} }` → `@theme inline {…}` (colours,
     fonts, radii) →
     `@layer base { * { @apply border-border outline-ring/50 } body { @apply bg-background text-foreground text-sm antialiased } button:not(:disabled), [role="button"]:not(:disabled) { cursor: pointer } }`
     → **then every existing legacy rule verbatim under `/* LEGACY — deleted in Phase 6 */`**
     so the app keeps its look while components migrate one by one. Remove the
     `@custom-variant dark` line.
  6. `index.html`: `<link rel="icon" type="image/svg+xml" href="/favicon.svg">`.
- **Verify:** typecheck/test/build/lint green;
  `ls src/agentic_erp_assistant/web/static/assets` shows one `.css`;
  `uv run agentic-erp-assistant serve` and the page looks unchanged, console clean
  (Tailwind preflight may shift default margins on `h1/h2/p/ul` — accept only differences
  the legacy block does not cover, or add the missing rule).
  Commit: `ui: add Tailwind v4 and shadcn tooling under the existing CSS`.

### Phase 2 — Foundation components and pure maps

- **Objective:** every shared building block exists and is tested before any screen changes.
- **Files:** `ui/src/components/ui/*` (shadcn add),
  `ui/src/components/shared/{SectionHeading,RouteBadge,ToneBadge,MonoId,KeyValueList,JsonBlock,EmptyState,ApprovalActions,ScopeChip,StreamingCaret}.tsx`,
  `ui/src/lib/{routeBadge,traceTone,format}.ts`, tests
  `ui/src/__tests__/{routeBadge,traceTone,format,ApprovalActions}.test.ts(x)`,
  `ui/src/setupTests.ts`.
- **Changes:**
  `npx shadcn@latest add -y button badge card select textarea separator tooltip table alert skeleton sheet collapsible`
  (+ `kbd empty spinner` if available). Move `routeBadge()` out of `AssistantMessage.tsx`
  into `lib/routeBadge.ts` (same branching, returns `{label, tone, icon}`); move
  `kindClass()` into `lib/traceTone.ts`. Write `format.ts`. In `setupTests.ts` add jsdom
  polyfills Radix needs: `Element.prototype.scrollIntoView`,
  `HTMLElement.prototype.hasPointerCapture/releasePointerCapture`, `ResizeObserver` stubs.
- **Verify:** new unit tests pin: every route-badge label string (they are quoted in
  `docs/manual-test.md §3`), tone for
  `route_selected/tool_called/approval_requested/memory_written/failed/run_failed`,
  `formatArgument({a:1}) === '{"a":1}'`, `formatCost(null) === 'unknown'`,
  `formatRelativeTime` boundaries. `ApprovalActions` test: both buttons disabled when
  `!canApprove` or `deciding`, names `Approve`/`Deny`. Lint may warn
  `only-export-components` on shadcn files exporting `*Variants` — allowed by
  `allowConstantExport`; add `"ignorePatterns": ["src/components/ui/**"]` to
  `.oxlintrc.json` only if it errors.
  Commit: `ui: shared design-system components and the pure badge/tone maps`.

### Phase 3 — App shell and sidebar

- **Objective:** the three-column frame, responsive drawers, and the whole left column on
  the new system.
- **Files:** `ui/src/App.tsx` (JSX only), `ui/src/components/layout/AppShell.tsx` (new),
  `ui/src/components/{ActorSwitcher,SessionList,ApprovalQueue}.tsx`.
- **Changes:** `AppShell` takes `sidebar`, `main`, `aside` nodes; owns `useState` for the
  two `Sheet`s (pure UI state). `App.tsx`: replace `<div className="app-layout">…` and the
  `Loading…` paragraph with `AppShell`; no other line changes. `ActorSwitcher`: `Select`
  with `onValueChange={onSelect}`, trigger `aria-label="Actor"`, item text
  `display_name — role`; scopes/`can approve`/`project_code` as chips. `SessionList`: rows
  → `<button>`, active styling, relative time, `EmptyState`. `ApprovalQueue`: cards +
  `ApprovalActions`, count badge, `EmptyState`. Delete the sidebar rules from the legacy
  CSS block.
- **Verify:** typecheck/test/build/lint; browser: six actors listed, switching actor
  clears chat (existing behaviour), remembered actor restored on reload, sessions load and
  highlight, approvals list shows and Approve/Deny fire from the queue as before; Tab/Enter
  reaches every session row; width 900px → drawers open/close, chat + composer visible.
  Commit: `ui: app shell with responsive drawers, sidebar on shadcn`.

### Phase 4 — Chat column

- **Objective:** messages, badges, citations, approval card, failure blocks, composer.
- **Files:**
  `ui/src/components/{ChatPanel,MessageList,AssistantMessage,CitationChips,FailureBlock,ApprovalCard,Composer}.tsx`,
  `ui/src/App.tsx` (pass `sessionId={state.sessionId}` — additive prop),
  `ui/src/__tests__/AssistantMessage.test.tsx` (only if a query needs a role/text tweak;
  the five existing assertions should still pass unchanged).
- **Changes:** as in §3 "Chat column". `AssistantMessage` imports `RouteBadge` (D4 rule
  unchanged: `turn.answer ? answer.text : streamed`). `ApprovalCard` uses `formatArgument`
  (fixes `[object Object]`). `MessageList` auto-scroll via bottom sentinel
  (`scrollIntoView?.()` guarded). Keep `aria-live="polite"` where it is. Delete chat rules
  from the legacy block.
- **Verify:** existing `AssistantMessage.test.tsx` green untouched (button names
  `Approve`/`Deny`, text `switch to an approver`, chip text `[m2-status.md#p.2]`, streamed
  text visible, answer replaces preview). Browser, per handbook: **R1** as `priya` — text
  streams with caret, is replaced by the final answer, chips render and open
  `/api/documents/…?actor=priya` in a new tab; **T1** — badge `tool`; **T6** as `tomas` —
  badge `failed` + destructive alert with `tool_failure`; **A-scenario** — card appears
  with arguments, enabled for `priya`, disabled with the hint for `tomas`; Stop mid-stream
  flips back to Send with no error; Enter sends, Shift+Enter inserts newline; list
  auto-scrolls after >15 messages and the composer never leaves the viewport.
  Commit: `ui: chat column on the design system; arguments render as JSON, not [object Object]`.

### Phase 5 — Trace panel

- **Objective:** the audit view reads as a dense, aligned table with a stable header and
  summary.
- **Files:** `ui/src/components/{TracePanel,TurnTrace,DecisionList,EventRow,TurnSummary}.tsx`,
  `ui/src/__tests__/EventRow.test.tsx`.
- **Changes:** as in §3 "Trace panel". `EventRow` renders a
  `<TableRow data-source={event.source}>`; keep the `·` placeholder for null seq. Update
  `EventRow.test.tsx` to assert `[data-source="tool_gateway"]` / `[data-source="engine"]`
  instead of `.trace-row.gateway` (the test coupled to a CSS class; the behaviour asserted
  is the same). `TurnTrace` still shows only the latest assistant message (existing
  behaviour). Delete trace rules from the legacy block.
- **Verify:** tests green; browser: **T11-style** gateway rows (or any
  `approval_requested` from the preflight) render italic/muted with no seq; decisions show
  arguments before the tool runs (T1); summary numbers match `/api/runs/<id>` (open via the
  header link — same URL as today); copy button puts the full `trace_id` on the clipboard;
  history-loaded turn shows the "no live trace" empty state with the run link.
  Commit: `ui: trace panel as a dense event table with decisions and a summary grid`.

### Phase 6 — Cleanup, polish, docs

- **Objective:** no legacy CSS, consistent states, handbook wording matches the screen.
- **Files:** `ui/src/index.css` (delete the LEGACY block), `ui/README.md` (replace
  boilerplate with a 20-line description of the app, scripts, and the protocol-drift
  rule), `ui/public/icons.svg` (delete), `ui/public/favicon.svg` (optional simple mark),
  `docs/manual-test.md` §3 (two phrasings: "the `<select>` in the sidebar" → "the actor
  selector", "Italic rows" → "Italic, muted rows marked ↳").
- **Changes:** grep `className="` for any surviving legacy class (`message`, `trace-row`,
  `scope-chip`, `approval-actions`, …) — zero hits; audit every interactive element for
  `focus-visible` ring and disabled styling; tooltips on truncated content; `Tooltip` on
  disabled Approve/Deny explaining why.
- **Verify:** full checklist in §6.
  Commit: `ui: remove legacy CSS, align the handbook's screen wording`.

---

## 5. Files touched (complete list)

| Kind | Path |
|---|---|
| **Modified** | `ui/package.json`, `ui/package-lock.json`, `ui/vite.config.ts`, `ui/tsconfig.json`, `ui/tsconfig.app.json`, `ui/index.html`, `ui/src/index.css`, `ui/src/setupTests.ts`, `ui/src/App.tsx` (JSX only), `ui/src/components/{ActorSwitcher,ApprovalCard,ApprovalQueue,AssistantMessage,ChatPanel,CitationChips,Composer,DecisionList,EventRow,FailureBlock,MessageList,SessionList,TracePanel,TurnSummary,TurnTrace}.tsx`, `ui/src/__tests__/EventRow.test.tsx`, `ui/README.md`, `docs/manual-test.md` (§3 wording), `ui/.oxlintrc.json` (only if needed) |
| **New** | `ui/components.json`, `ui/src/lib/{utils,routeBadge,traceTone,format}.ts`, `ui/src/components/ui/*` (shadcn), `ui/src/components/shared/*` (10 files), `ui/src/components/layout/AppShell.tsx`, `ui/src/__tests__/{routeBadge,traceTone,format,ApprovalActions}.test.*` |
| **Deleted** | `ui/public/icons.svg` |
| **Never touched** | `ui/src/{protocol,turnReducer,appState,useTurnStream,sse,api}.ts`, `ui/src/__tests__/{turnReducer,sse}.test.ts`, `ui/src/main.tsx`, anything under `src/agentic_erp_assistant/`, `tests/` |

---

## 6. Implementation Order

0 baseline → 1 tooling (app unchanged) → 2 shared components + pure maps (tests first) →
3 shell + sidebar → 4 chat → 5 trace → 6 cleanup. Phases 3, 4, 5 are independent of each
other once 2 is done, but doing them in that order keeps the legacy CSS block shrinking
monotonically. One commit per phase on `fpt-bao/refactor-UI`; each commit leaves the app
runnable.

**Final verification (end of Phase 6, all mandatory):**

1. `npm --prefix ui run typecheck && npm --prefix ui test && npm --prefix ui run build && npm --prefix ui run lint`.
2. `uv run pytest -q` (protocol drift test proves `protocol.ts` is untouched).
3. `uv run agentic-erp-assistant serve`, then handbook scenarios **S1, S3, S4, R1, R11, T1,
   T6, A1 (approve from card), A1 (deny from queue as `sponsor`), T9 (session history
   reload)**; browser console clean throughout.
4. DevTools: emulate `prefers-color-scheme: dark` — every tone badge/alert readable;
   viewport 400px and 900px — chat and composer usable, drawers work.
5. Bundle: `assets/*.js` gzip ≲ 200 kB (Radix + Lucide tree-shaken); no external requests
   except `/api/*`.

---

## 7. Risk / Scope Notes

- **Hard boundary:** `App.tsx` handlers, effects, and reducer wiring stay line-for-line;
  component **prop signatures are unchanged except additive optional props**
  (`ChatPanel.sessionId`). No new fetches (health, run JSON stay as links), no routing, no
  persistence changes.
- **Tests coupled to markup:** `EventRow.test.tsx` asserts CSS classes → migrates to
  `data-source`. `AssistantMessage.test.tsx` asserts roles/text → must keep button text
  `Approve`/`Deny`, hint text `switch to an approver…`, chip text exactly the tag, Lucide
  icons `aria-hidden`.
- **Radix in jsdom:** `Select`/`Tooltip`/`Sheet` need the `setupTests.ts` polyfills;
  without them any new sidebar test throws on `hasPointerCapture`/`scrollIntoView`.
- **Radix `Select` is not a native `<select>`:** the handbook wording changes; no
  functional difference (keyboard, typeahead work). If any Python/browser automation
  targets `select[aria-label=Actor]`, it would break — none found in `tests/` or `scripts/`.
- **Scrolling regression risk (ADR 0015 documents this exact bug):** keep `min-h-0` on
  every flex/grid child that scrolls; do **not** wrap the message list in Radix
  `ScrollArea` (its viewport needs the same care and adds nothing).
- **`shadcn init` rewrites `index.css`:** commit before running it, diff after, and
  re-insert the legacy block; the plan relies on the legacy rules surviving until Phase 6.
- **Dark mode without a toggle** is deliberate (parity with today). A theme toggle is a
  follow-up, not part of this plan.
- **Known data limits the UI must not paper over:** history-loaded turns carry no
  citations or trace (`SessionTurn` has neither) and a paused history turn has no approval
  card (decide from the queue) — render honest empty states, do not synthesise.
  `/api/users` failure still shows the loading state; surfacing an error card would need
  one local state variable in `App.tsx` — left out as it brushes the "no state changes"
  rule; flag for sign-off if wanted.
- **Dependencies added** (all required by the requested stack): `tailwindcss`,
  `@tailwindcss/vite`, `class-variance-authority`, `clsx`, `tailwind-merge`,
  `lucide-react`, `tw-animate-css`, Radix primitives (`radix-ui` or
  `@radix-ui/react-{select,tooltip,dialog,collapsible,separator,slot}`). Nothing else; no
  fonts, no icons from the network.
- Bundle grows (~+80–120 kB gzip); acceptable for a served-by-FastAPI dev tool.
