# WEEK 1 EXECUTION GUARANTEE
**faster-blaster Phase 2 Launch - Success Formula**

**Document Owner:** Tech Lead  
**Created:** Jan 27, 2026  
**Team Size:** 3 developers + 1 QA + 1 Tech Lead  
**Operations This Week:** 4 (SAXPY ✓ → DAXPY → CAXPY → ZAXPY)  
**Status:** 🟢 READY TO EXECUTE

---

## THE GUARANTEE

**If we follow this process exactly, Week 1 is 100% guaranteed to deliver:**
- ✅ 4 operations fully implemented
- ✅ 32+ comprehensive tests (all passing)
- ✅ Zero memory leaks (Valgrind clean)
- ✅ Code review approved
- ✅ Team momentum for 50 ops/week scaling

**This document is the single source of truth for Week 1 success.**

---

## PHASE 1: PREPARATION (Friday Jan 26 - Completed)

### What We've Done
- ✅ Analyzed BLIS source code (44 lines = SAXPY complexity baseline)
- ✅ Created test template (8-test pattern covers all edge cases)
- ✅ Built tracking spreadsheet (real-time visibility)
- ✅ Wrote detailed kickoff guide (Monday agenda)
- ✅ Prepared code review checklist (consistent standards)

### What Each Team Member Has
1. **Dev 1 (SAXPY/DAXPY):** Reference implementation + template
2. **Dev 2 (CAXPY):** Complex number reference + example
3. **Dev 3 (ZAXPY):** Double complex reference + formula sheet
4. **QA:** Test template + Valgrind validation guide
5. **Tech Lead:** Code review checklist + blocker resolution protocol

**Nobody is starting blind.** Everything is prepped.

---

## PHASE 2: MONDAY KICKOFF (Jan 27, 1:00-5:00 PM)

### Purpose
Get everyone on same page, answer every question before coding starts.

### Agenda (Exact - 4 hours)

**1:00-1:15 PM: Opening (15 min)**
- Goal: 4 operations, 32 tests, Friday EOD
- Success metric: All tests pass + Valgrind clean
- Pace: 1 op/day baseline, faster if team syncs well
- Tone: "We have everything prepared. Let's execute."

**1:15-1:45 PM: BLIS Code Walkthrough (30 min)**
- **Tech Lead reads SAXPY code line-by-line** (44 lines)
- Explains: Stride logic, loop structure, alpha scaling
- Pause: Anyone unclear? Ask now (not Tuesday)
- Key insight: "This is the pattern for 54 BLIS L1 ops"

**1:45-2:15 PM: C23 Complex Numbers Deep Dive (30 min)**
- **Whiteboard:**
  ```
  float _Complex z = 1.0f + 2.0f*I;
  float real_part = crealf(z);       // 1.0
  float imag_part = cimagf(z);       // 2.0
  
  Complex multiplication:
  (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
  
  Example: (1+2i) * (3+4i)
  = (1*3 - 2*4) + (1*4 + 2*3)i
  = -5 + 10i
  ```
- Reference: ISO C99 Annex G (complex arithmetic)
- Practice: Dev 2 writes example on board

**2:15-2:45 PM: Test Template Walkthrough (30 min)**
- **QA presents 8-test pattern:**
  1. Basic (stride=1)
  2. Non-contiguous (stride=2)
  3. Empty (n=0)
  4. Zero alpha (α=0)
  5. Single element (n=1)
  6. Negative alpha (α<0)
  7. Negative stride (backward)
  8. Large values (precision check)
- Question: "Which test catches what bug?"
- Everyone agrees: These 8 tests = confident op is correct

**2:45-3:15 PM: Tools & Environment Setup (30 min)**
- **Build command:** `cd build && cmake --build . -j4`
- **Run test:** `ctest --verbose --output-on-failure`
- **Valgrind command:** `valgrind --leak-check=full ./test_blis_l1_saxpy`
- **Spreadsheet access:** All 5 people can edit (GitHub or shared drive)
- **Code review:** Submit to TL via PR or direct file
- **Slack/Chat:** For blockers (1-hour response guarantee)

**3:15-3:45 PM: Standup Demo (30 min)**
- **Tech Lead demonstrates:**
  - Morning standup format (4 min per person)
  - Status update structure: Done | Doing | Blocked
  - How to ask for help
  - Spreadsheet update procedure
- **Each team member rehearses 30 seconds**

**3:45-4:00 PM: Final Q&A (15 min)**
- Open questions about anything
- Tech Lead clarifies or reschedules 1:1
- Goal: Everyone leaves saying "I'm ready for Tuesday"

**4:00-4:15 PM: Close (15 min)**
- **Confirm:** "Tuesday 9 AM, everyone ready?"
- **Announce:** All 5 team members sign the "Week 1 Commitment"

### Commitment Statement (All sign)
> "I have understood the requirements, reviewed the reference code, and am committed to delivering my assigned operations with comprehensive tests by Friday EOD. I will communicate blockers immediately and support my team. Let's execute Week 1."

**No one codes until this is done.**

---

## PHASE 3: EXECUTION DAYS (Tue-Thu, Jan 28-30)

### Daily Schedule (Same Each Day)

**9:00-9:15 AM: Standup**
- **Each person (4 min total):**
  - "Yesterday: [what's done]"
  - "Today: [what's next]"
  - "Blocked?: [yes/no/what]"
- **Tech Lead action items:** Who needs help? When?
- **Spreadsheet updated:** Before this meeting

**9:15 AM - 12:30 PM: Development Sprint 1 (3.25 hours)**
- **Dev 1, 2, 3:** Code their assigned operation
- **QA:** Write/refine tests in parallel
- **Tech Lead:** Available for blockers (quick help = 15 min max)

**12:30-1:30 PM: Lunch Break**

**1:30-5:00 PM: Development Sprint 2 (3.5 hours)**
- **Dev 1, 2, 3:** Continue coding + build testing
- **QA:** Run tests as code becomes available
- **Tech Lead:** Code reviews (aim for 2x per person, per day)

**5:00 PM: EOD Status**
- Each person: Update spreadsheet
- Tech Lead: Review today's progress, plan tomorrow

### Development Workflow

**Dev 1 (Tuesday-Wednesday):**

| Time        | Task                              | Expected Output      |
| ----------- | --------------------------------- | -------------------- |
| Tue 9-12:30 | Verify SAXPY exists (should be ✓) | Confirmed            |
| Tue 1:30-5  | Copy SAXPY → DAXPY, modify types  | `daxpy.c` created    |
| Wed 9-12:30 | Debug build errors                | Compiles             |
| Wed 1:30-3  | Test with QA tests                | 6/8 passing          |
| Wed 3-5     | Fix failing tests                 | 8/8 passing          |
| Wed EOD     | Code review ready                 | TL reviews Wed night |

**Dev 2 (Wednesday-Thursday):**

| Time        | Task                                                    | Expected Output         |
| ----------- | ------------------------------------------------------- | ----------------------- |
| Tue 9-5     | Read complex number reference                           | Understanding confirmed |
| Wed 9-12:30 | Copy DAXPY (Dev 1's version) → CAXPY, modify to complex | `caxpy.c` created       |
| Wed 1:30-5  | Debug complex type casting                              | Compiles                |
| Thu 9-12:30 | Test with QA tests                                      | 6/8 passing             |
| Thu 1:30-3  | Fix precision/tolerance issues                          | 8/8 passing             |
| Thu 3-5     | Valgrind validation                                     | Clean                   |
| Thu EOD     | Code review ready                                       | TL reviews Thu night    |

**Dev 3 (Thursday):**

| Time        | Task                                         | Expected Output      |
| ----------- | -------------------------------------------- | -------------------- |
| Tue-Wed     | Review double complex math                   | Ready to code        |
| Thu 9-12:30 | Copy CAXPY → ZAXPY, modify to double complex | `zaxpy.c` created    |
| Thu 1:30-3  | Test tolerance (1e-15 vs 1e-6)               | 6/8 passing          |
| Thu 3-5     | Final fixes + Valgrind                       | 8/8 passing          |
| Thu EOD     | Code review ready                            | TL reviews Thu night |

**QA (Parallel):**

| Time        | Task                                    | Expected Output   |
| ----------- | --------------------------------------- | ----------------- |
| Tue 9-12:30 | Finalize SAXPY tests (should pass)      | 8/8 ✓             |
| Tue 1:30-5  | Create DAXPY tests from template        | 8 tests ready     |
| Wed 9-5     | Create CAXPY tests, adjust tolerance    | 8 tests ready     |
| Thu 9-5     | Create ZAXPY tests, adjust tolerance    | 8 tests ready     |
| All days    | Run tests as they come, report failures | Daily test report |

### Critical Rules for Execution

**RULE 1: No "perfect" code in first pass**
- Write something → test it → fix it → review it
- Perfectionism delays by days
- "Done" = tests pass, Valgrind clean, code reviewed

**RULE 2: Ask blockers immediately**
- Don't spend 30 min debugging alone
- Mention in standup → TL helps within 15 min
- Better to get help than to stall

**RULE 3: Test as you go**
- Build daily (don't wait until Thursday)
- Run tests daily (QA will report failures)
- Fix failures same day (not Friday)

**RULE 4: Code review before Friday**
- Wed for Dev 1/DAXPY
- Wed-Thu for Dev 2/CAXPY
- Thu for Dev 3/ZAXPY
- Don't wait until Friday (no time for fixes)

**RULE 5: Update spreadsheet daily**
- Before standup each morning
- Actual % complete (not optimistic)
- Blockers noted (so TL can prep help)

---

## PHASE 4: VALIDATION CHECKS (Per Operation)

### Before Marking "Complete" (Checklist)

**When Dev finishes coding:**
1. ✅ Code compiles: `cmake --build .`
2. ✅ No errors/warnings printed
3. ✅ Follows SAXPY pattern (stride logic, early exits)
4. ✅ Comments explain key steps
5. ✅ Submitted for code review

**When QA runs tests:**
1. ✅ 8 tests written (at least)
2. ✅ 8/8 tests passing
3. ✅ Output shows PASS for each
4. ✅ Test tolerances correct (float=1e-6, double=1e-15)
5. ✅ Edge cases covered (n=0, alpha=0, negative stride)

**After code review approved:**
1. ✅ TL signed off (PR merge or email)
2. ✅ Zero build warnings
3. ✅ Valgrind clean: `valgrind --leak-check=full ./test ← zero errors`
4. ✅ Spreadsheet marked: `status = "COMPLETE"`
5. ✅ Operation ready for next week

**If any check fails:**
- Dev fixes it (not QA's job)
- Re-submit for review
- Aim for same-day turnaround (don't wait until next day)

---

## PHASE 5: FRIDAY RETROSPECTIVE (Jan 31, 10:00 AM-12:30 PM)

### Morning Standup (9:00 AM - Quick)
- Each person: 2-min status
- Expected: All 4 ops "COMPLETE"
- If not: What's the blocker? How do we fix it Fri morning?

### Main Retrospective (10:00-11:30 AM - 90 min)

**Part 1: Metrics Review (20 min)**
- ✅ Operations completed: Target 4, actual: ?
- ✅ Tests written: Target 32, actual: ?
- ✅ Tests passing: Target 100%, actual: ?
- ✅ Valgrind: Target clean, actual: ?
- ✅ Code reviews: All approved? Yes/No
- ✅ Velocity: 4 ops in 4 days = 1 op/day baseline confirmed?

**Part 2: Team Feedback (40 min)**
- **What went well?** (10 min)
  - What process helped us move fast?
  - What tool/template saved time?
  - What communication worked?
- **What was hard?** (10 min)
  - What slowed us down?
  - Did any tool fail us?
  - Where did we help each other?
- **What do we change for Week 2?** (10 min)
  - Keep or discard? (standups, templates, tracking)
  - Faster pace needed? (50 ops/week needs what?)
  - New tools or processes?
- **Team celebration** (10 min)
  - Week 1 complete = achievement!
  - Shout-outs to strong performers
  - Preview: Week 2 will be 10x faster with this momentum

**Part 3: Week 2 Planning (30 min)**
- **4 new operations:** SCAL, DSCAL, CSCAL, ZDSCAL (scaling)
- **Same team?** (Yes - consistency helps)
- **Same process?** (Yes - but maybe faster)
- **Pace increase:** 4 ops (Week 1) → 50 ops/week target
- **Estimated timeline:** 50 ops = 2.5 weeks for Phase 2 BLAS L1-L3
- **Confidence level:** "Ready to scale?" (vote: all say yes = GO)

### Success Criteria for Friday

**To "WIN" Week 1:**

- [ ] SAXPY: 100% (already verified) ✓
- [ ] DAXPY: 100%, 8 tests pass, Valgrind clean
- [ ] CAXPY: 100%, 8 tests pass, Valgrind clean
- [ ] ZAXPY: 100%, 8 tests pass, Valgrind clean
- [ ] Total: 4 ops, 32 tests, zero memory issues
- [ ] Code reviews: All approved (no PRs still pending)
- [ ] Spreadsheet: 100% up-to-date
- [ ] Team readiness: Confidence to scale to 50 ops/week
- [ ] Documentation: Week 1 retrospective + findings written

**If all checked:** Week 1 = PERFECT LAUNCH ✅

**If any unchecked:**
- Identify root cause (process issue? blocker? tooling?)
- Fix for Week 2 (don't ignore)
- Adjust timeline (if velocity lower than 4 ops/week, reduce targets)

---

## THE BLOCKER RESOLUTION PROTOCOL

**If you get stuck at any point this week:**

### Level 1: Self-Help (2 minutes)
- Check error message (read it carefully)
- Google error + your compiler (e.g., "float _Complex gcc error")
- Look at reference code (SAXPY for your operation)
- Reread relevant documentation

### Level 2: Team Help (10 minutes)
- Ask in Slack/chat channel (don't wait for standup)
- Another dev (1-2 min) or QA might have answer
- Example: "How do I create float _Complex in C?"

### Level 3: Tech Lead Help (15 minutes)
- Mention in standup: "I'm blocked on [X]"
- TL responds: "Let's debug together, 15 min"
- **TL takes ownership** until unblocked
- Tech Lead can:
  - Explain code pattern
  - Debug compiler error
  - Pair program (screen share)
  - Adjust task scope (if unrealistic)

### Level 4: Escalation (>1 hour blocked)
- TL reports to PM: "Team blocked on [X], needs help"
- PM either:
  - Provides external expertise (consultant call)
  - Adjusts scope (reduce target for this op)
  - Extends timeline (accept slower pace)
- **Goal:** Nobody is stuck for more than 1 hour

**Important:** This escalation has happened 0 times in Phase 2 prep. We're confident Level 1-3 handle everything.

---

## WHAT SUCCESS LOOKS LIKE

### By Friday EOD Week 1

**In code:**
- 4 new C files: `daxpy.c`, `caxpy.c`, `zaxpy.c` (saxpy.c already exists)
- Each: 30-50 lines, follows BLIS pattern, compiles cleanly
- Each: Implements correct operation (stride logic, alpha scaling)

**In tests:**
- 32 test functions (8 per operation)
- All pass (green checkmarks in `ctest`)
- Valgrind clean (zero leaks/errors reported)
- Coverage: Basic case + edge cases + negative values

**In tracking:**
- Spreadsheet 100% complete
- Every row has status, test count, approval mark
- Comments capture what was easy vs. hard

**In team:**
- All 5 people finished Friday afternoon
- Confidence level: "We can do 50 ops/week"
- No team members burned out (all energized for Week 2)
- Retrospective identified improvements for scaling

**In process:**
- Standup format proven (9:00-9:15 effective)
- Code review checklist worked (caught issues early)
- Test template scaled (same 8-test pattern for 4 ops)
- Tracking spreadsheet provided real-time visibility

---

## WHAT FAILURE LOOKS LIKE (Avoid!)

❌ **Monday:** Kickoff not done (people still unclear Tuesday)  
❌ **Tuesday:** Dev starts coding without reference, gets stuck on type casting  
❌ **Wednesday:** Test failures not fixed (wait until Thursday = no review time)  
❌ **Thursday:** Code review not done (Friday morning = no time to fix issues)  
❌ **Friday:** 2 ops done, 2 still "in progress" (team demoralized)  
❌ **Valgrind:** Memory leaks found Friday afternoon (no time to debug)  
❌ **Spreadsheet:** Not updated (nobody knows actual progress)  
❌ **Team:** Burnout from thrashing (next week is slower, not faster)

**We avoid these by following the schedule exactly.**

---

## THE COMMITMENT

### All 5 Team Members Sign:

**"I commit to Week 1 execution guarantee:**
- **I will attend all standups** (9:00 AM sharp, Tue-Fri)
- **I will update the tracking spreadsheet daily** (before standup)
- **I will follow the development schedule** (not just "whenever")
- **I will communicate blockers immediately** (not wait until Friday)
- **I will support my team** (help others if I finish early)
- **I will deliver my assigned operations complete** (code + tests + reviews)
- **I will prepare for Week 2 scaling** (learning from Week 1 to do 50 ops/week)

Signed: ____________________ (Date: Jan 27, 2026)"

---

## FINAL CHECKLIST BEFORE MONDAY 1 PM KICKOFF

**Tech Lead verifies:**
- ✅ All 5 team members received this document
- ✅ SAXPY reference code confirmed (44 lines)
- ✅ Test template prepared (8-test pattern)
- ✅ Spreadsheet created and shared (all can edit)
- ✅ Slack/chat channel ready (real-time communication)
- ✅ Code review process documented (TL knows how to review)
- ✅ Build environment tested (cmake works)
- ✅ Valgrind installed (if Linux/WSL)
- ✅ All team members confirm readiness (email/Slack)
- ✅ Meeting room booked (1:00-5:00 PM Monday)

**Go/No-Go Decision (Sunday Jan 26):**
- If all checked: "GO for Week 1 kickoff Monday"
- If any not checked: "Fix it Sunday before Monday"

---

## THE SUCCESS FORMULA

**Formula: Preparation + Clear schedule + Daily sync + Immediate blocker help = Success**

**Week 1 is guaranteed to succeed if we:**
1. ✅ Do the 4-hour Monday kickoff (answer all questions now)
2. ✅ Code Tue-Wed-Thu (not "squeeze it in")
3. ✅ Test daily (not Friday-before-deadline)
4. ✅ Review before Friday (not Friday afternoon)
5. ✅ Unblock immediately (not let people spin)
6. ✅ Track in real-time (not guessing on Friday)

**No excuses. No shortcuts. This process works.**

---

## WEEK 1 COMMITMENT: READY TO EXECUTE 🚀

**By Monday 1 PM, everyone says: "Let's go."**

**By Friday 5 PM, everyone celebrates: "Week 1 complete."**

**By Monday Feb 3, everyone scales: "Week 2 at 50 ops/week."**

---

**Document Version:** 1.0  
**Last Updated:** Jan 26, 2026  
**Status:** 🟢 READY FOR EXECUTION  
**Next Review:** Friday Jan 31 (retrospective)

---

*"Execution is 95% discipline, 5% inspiration. This guide provides the discipline. The team provides the inspiration. Together = success."*

---

**🎯 See you Monday 1 PM. Let's launch faster-blaster Phase 2. 🎯**
