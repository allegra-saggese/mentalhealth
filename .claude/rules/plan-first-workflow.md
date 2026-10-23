# Plan-First Workflow

**Enter plan mode before tasks that touch more than three files or change a deliverable's substance** (a lecture, a manuscript, an analysis pipeline, a replication package). Small, well-specified edits proceed directly — planning them costs more than it saves.

## The Protocol

1. **Enter Plan Mode** — use `EnterPlanMode`
2. **Check MEMORY.md** — read any `[LEARN]` entries relevant to this task
3. **Requirements Specification (for complex/ambiguous tasks)** — see below
4. **Draft the plan** — what changes, which files, in what order
5. **Save to disk** — write to `quality_reports/plans/YYYY-MM-DD_short-description.md`
6. **Present to user** — wait for approval
7. **Exit plan mode** — only after approval
8. **Save initial session log** — capture goal and key context while fresh
9. **Implement via orchestrator** — see `orchestrator-protocol.md`

## Step 3: Requirements Specification (For Complex/Ambiguous Tasks)

**When to use:**
- Task is high-level or vague ("improve the lecture", "analyze the data")
- Multiple valid interpretations exist
- Significant effort required (>1 hour or >3 files)

**When to skip:**
- Task is clear and specific ("fix typo in line 42")
- Simple single-file edit
- User has already provided detailed requirements

**Protocol:**
1. Use AskUserQuestion to clarify ambiguities (max 3-5 questions)
2. Create `quality_reports/specs/YYYY-MM-DD_description.md` using `templates/requirements-spec.md`
3. Mark each requirement:
   - **MUST** (non-negotiable)
   - **SHOULD** (preferred)
   - **MAY** (optional)
4. Declare clarity status for each major aspect:
   - **CLEAR:** Fully specified
   - **ASSUMED:** Reasonable assumption (user can override)
   - **BLOCKED:** Cannot proceed until answered
5. Get user approval on spec
6. THEN proceed to Step 4 (draft the plan) with spec as input

**Template:** `templates/requirements-spec.md`

**Why this helps:** Catches ambiguity BEFORE planning. Reduces mid-plan pivots by 30-50%.

## Plans on Disk

Plans survive context compression. Save every plan to:

```
quality_reports/plans/YYYY-MM-DD_short-description.md
```

Format: Status (DRAFT/APPROVED/COMPLETED), approach, files to modify, verification steps.

## Context Management

### General Principles
- Within a task, let compaction carry you — steer it with `/compact <what to keep>`
- Between unrelated tasks, `/clear`
- Save important context to disk before it's lost
- To stop or hand off: `/checkpoint`, quit, then start a fresh `claude` — the session-handoff
  hook hands the newest checkpoint, once, to the next fresh start (not to `--continue` /
  `--resume`); after that, ask Claude to read the checkpoint file

### Context Survival Strategy

**Before Auto-Compression:**
When approaching context limits, ensure:
1. MEMORY.md has all `[LEARN]` entries from this session
2. Session log is current (updated within 10 minutes)
3. Active plan is saved to disk
4. Open questions are documented in session log

The compaction hooks carry state across on their own: `pre-compact.py` saves the active plan, the current task, and recent decisions, and `post-compact-restore.py` re-injects them afterwards. They can only restore what is on disk — keeping the plan and session log current is what makes the restored state accurate.

**After Compression:**
First message should be: "Resuming after compression. Last task: [read most recent plan + git log]. Status: [next step]."

## Session Recovery

After compression or new session:
1. Read `CLAUDE.md` + most recent plan in `quality_reports/plans/`
2. Check `git log --oneline -10` and `git diff`
3. State what you understand the current task to be
