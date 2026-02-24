# Week 1 Quick Start Reference
**faster-blaster Phase 2 - At-a-Glance Guide**  
**Print this or bookmark for quick lookup**

---

## 🎯 THIS WEEK (Jan 27-31)

**4 Operations:** SAXPY (verify) → DAXPY → CAXPY → ZAXPY  
**32+ Tests:** 8 per operation (edge cases covered)  
**Timeline:** Mon kickoff + Tue-Thu coding + Fri retrospective  
**Success:** 100% tests passing + Valgrind clean + code review approved

---

## 📋 DAILY SCHEDULE

### Monday (Jan 27) - Kickoff Only
- **1:00 PM**: Start video conference
- **1:00-5:00 PM**: 4-hour kickoff (see WEEK1_TEAM_KICKOFF_GUIDE.md)
- **5:00 PM**: Team confirms "Ready for Tuesday 9 AM"

### Tuesday-Thursday (Jan 28-30) - Coding Days
- **9:00-9:15 AM**: Daily standup (each person: 4 min)
- **9:15 AM-5:00 PM**: Development, testing, code review
- **EOD**: Update tracking spreadsheet

### Friday (Jan 31) - Retrospective
- **9:00 AM**: Final standup
- **10:00-11:30 AM**: Retrospective (what worked, what to improve)
- **11:30 AM-12:30 PM**: Week 2 planning
- **End of day**: Celebration if 4 ops complete ✅

---

## 🔧 DEVELOPER QUICK START

### What to Do (Your First Day - Tuesday 9 AM)

**Dev 1 (SAXPY verify + DAXPY):**
1. Open: `src/blis/level1/saxpy.c` (44 lines)
2. Read: Understand stride logic (if incx==1 fast path, else general path)
3. Copy: `saxpy.c` → `daxpy.c`
4. Edit: Replace `float` with `double`, `float *x` with `double *x`, etc.
5. Test: Run `test_blas_l1_daxpy.c` (QA will create)
6. Review: By 3 PM Tuesday

**Dev 2 (CAXPY with complex):**
1. Read: About C23 `float _Complex` type (15 min)
2. Copy: `daxpy.c` (Dev 1's version) → `caxpy.c`
3. Edit: Replace `double` with `float _Complex`
4. Reference: `a + bi = (a, b)` as real + imag parts
5. Test: Run `test_blas_l1_caxpy.c` (QA will create)
6. Review: By 3 PM Wednesday

**Dev 3 (ZAXPY with double complex):**
1. Copy: `caxpy.c` (Dev 2's version) → `zaxpy.c`
2. Edit: Replace `float _Complex` with `double _Complex`
3. Copy-paste: Test tolerance becomes 1e-15 (vs 1e-6)
4. Test: Run `test_blas_l1_zaxpy.c` (QA will create)
5. Review: By 3 PM Thursday

**All Developers:**
- Build command: `cd build && cmake --build . -j4`
- Your code location: `src/blis/level1/*.c` files
- Test location: `tests/test_blis_l1_*.c` files
- Update spreadsheet: Fill `impl_percent` and `tests_written` columns daily
- If stuck: Mention in standup, TL will help within 1 hour

---

## 🧪 QA QUICK START

### What to Do (Your First Day - Tuesday 9 AM)

1. **Get the template:**
   - Open: `tests/test_blis_l1_saxpy.c` (230 lines)
   - Read: Understand the 8-test pattern

2. **Understand the 8 tests:**
   - Test 1: Basic operation (arrays contiguous, stride=1)
   - Test 2: Non-contiguous (stride=2, skip every other)
   - Test 3: Edge case: n=0 (array unchanged)
   - Test 4: Edge case: alpha=0 (array unchanged)
   - Test 5: Edge case: n=1 (single element)
   - Test 6: Negative alpha (subtraction instead of addition)
   - Test 7: Negative stride (backward iteration)
   - Test 8: Large values (test precision, no overflow)

3. **Create tests (parallel with dev work):**
   - Copy template for SAXPY → create DAXPY tests
   - Copy template for DAXPY → create CAXPY tests (complex tolerance)
   - Copy template for CAXPY → create ZAXPY tests (double complex tolerance)

4. **Run tests:**
   - Command: `cd build && ctest --verbose`
   - Expected: All tests pass (green checkmarks)
   - Failures: Investigate within 1 hour, TL helps

5. **Valgrind validation (Linux/WSL):**
   - Command: `valgrind --leak-check=full ./test_blis_l1_saxpy`
   - Expected: "ERROR SUMMARY: 0 errors"
   - If issues: Document and report to TL

6. **Update spreadsheet:**
   - `tests_written` column: How many tests created
   - `tests_passing` column: How many tests pass
   - Daily update (Tue-Fri before standup)

---

## 💻 TECH LEAD QUICK START

### What to Do (Your First Day - Monday PM kickoff)

1. **Prepare code review checklist** (before Tue 9 AM):
   - Code compiles without errors
   - Stride logic correct (matches SAXPY pattern)
   - Early exits handled (n<=0, alpha==0)
   - Comments clear
   - Tests > 0 (at least trying)

2. **Lead standup** (9:00-9:15 AM each day):
   - Round-robin: Each person gives 4-min update
   - Capture: What's done, what's blockers
   - Decide: Who needs help today

3. **Code reviews** (as implementations arrive):
   - SAXPY: Verify by Tue 2 PM
   - DAXPY: Verify by Tue 4 PM
   - CAXPY: Verify by Wed 3 PM
   - ZAXPY: Verify by Thu 3 PM
   - If issues: Send back with specific comments, 1-hour turnaround for fixes

4. **Unblock issues** (real-time during implementation):
   - Blocked on C23 complex? Give example code
   - Build fails? Debug with dev in real-time
   - Valgrind error? Help investigate
   - Target: No blocker lasts >1 hour

5. **Friday retrospective** (10:00 AM - 12:30 PM):
   - Verify: 4 ops complete, 32 tests pass, Valgrind clean
   - Review: What went well, what to improve
   - Metrics: Actual velocity vs. 4 ops/week target
   - Decision: GO for Week 2 scaling (50 ops/week)

---

## 🗂️ CRITICAL FILE LOCATIONS

```
faster-blaster/
├── src/blis/level1/
│   ├── saxpy.c           ← Reference (read this first!)
│   ├── daxpy.c           ← You create
│   ├── caxpy.c           ← You create
│   └── zaxpy.c           ← You create
├── tests/
│   ├── test_blis_l1_saxpy.c   ← Template (8 tests)
│   ├── test_blis_l1_daxpy.c   ← Copy template + adapt
│   ├── test_blis_l1_caxpy.c   ← Copy template + adapt
│   └── test_blis_l1_zaxpy.c   ← Copy template + adapt
├── build/
│   └── (your build directory - run cmake here)
├── MASTER_IMPLEMENTATION_SPREADSHEET_TIER1.csv  ← Tracking
├── WEEK1_TEAM_KICKOFF_GUIDE.md                  ← Monday procedures
└── WEEK1_DAILY_STANDUP_TEMPLATE.md              ← Notes each day
```

---

## 📊 TRACKING SPREADSHEET COLUMNS

**Update daily (before standup):**

| Column             | Meaning               | Example               |
| ------------------ | --------------------- | --------------------- |
| operation_name     | What you're coding    | DAXPY                 |
| owner              | Who's responsible     | Dev 1                 |
| impl_percent       | % implementation done | 50                    |
| tests_written      | Tests created         | 8                     |
| tests_passing      | Tests passing         | 6                     |
| code_review_status | Code review state     | In Progress           |
| notes              | Any blockers/comments | Waiting for TL review |

**Example Monday EOD:**
- SAXPY: Dev1, 10%, 0 tests, 0 passing, Verified, "baseline checked"
- DAXPY: Dev1, 0%, 0 tests, 0 passing, Not Started, "starting Tuesday"

**Example Friday EOD:**
- SAXPY: Dev1, 100%, 8 tests, 8 passing, Approved, "complete ✓"
- DAXPY: Dev1, 100%, 8 tests, 8 passing, Approved, "complete ✓"
- CAXPY: Dev2, 100%, 8 tests, 8 passing, Approved, "complete ✓"
- ZAXPY: Dev3, 100%, 8 tests, 8 passing, Approved, "complete ✓"

---

## 🆘 IF YOU GET STUCK

### "I don't understand the stride logic"
- **Solution**: Read SAXPY code (44 lines) + comments
- **Formula**: `int ix = (incx > 0) ? 0 : (n-1)*(-incx);`
- **Ask**: TL in standup or chat

### "C23 complex multiplication is confusing"
- **Formula**: `(a + bi) * (c + di) = (ac-bd) + (ad+bc)i`
- **In code**: `a_real*b_real - a_imag*b_imag` + `i*(a_real*b_imag + a_imag*b_real)`
- **Example**: `(1 + 2i) * (3 + 4i) = (1*3 - 2*4) + i*(1*4 + 2*3) = -5 + 10i`
- **Ask**: Dev 2 or TL

### "Build fails"
- **First try**: `cd build && cmake --build . -j4 --verbose`
- **Check**: Do you have a missing `#include` or typo?
- **Ask**: TL (they can debug within 15 min)

### "Test fails"
- **First try**: Run test manually to see output
- **Check**: Did you update tolerance for double vs float?
- **Ask**: QA or TL to investigate

### "Valgrind shows memory leak"
- **Possible**: You're not freeing allocated memory
- **Check**: Did you `malloc()` anywhere? Did you `free()` it?
- **Ask**: TL (memory debugging expertise)

### "I have a blocker"
- **Mention in standup**: 9:00 AM (don't wait all day)
- **Get help**: TL will take ownership immediately
- **Escalate**: If >1 hour, TL escalates to PM

---

## ✅ END-OF-DAY CHECKLIST (Each person - 5 min)

**Every day Tue-Fri at 5 PM:**

- [ ] Code compiles (no errors)
- [ ] My part of spreadsheet updated
- [ ] If I'm blocked: Mentioned in today's standup
- [ ] If code review ready: TL has it
- [ ] Ready for tomorrow standup (9:00 AM)

---

## 🎉 FRIDAY SUCCESS CHECKLIST

**End of Week 1 (Friday 5 PM):**

- [ ] SAXPY: 100% complete ✓
- [ ] DAXPY: 100% complete ✓
- [ ] CAXPY: 100% complete ✓
- [ ] ZAXPY: 100% complete ✓
- [ ] Tests: 32+ all passing ✓
- [ ] Valgrind: Clean (zero leaks/errors) ✓
- [ ] Code reviews: All approved ✓
- [ ] Spreadsheet: 100% up-to-date ✓
- [ ] Team: Ready for Week 2 ✓

**If all checked:** Week 1 = SUCCESS 🎊

---

## 🚀 WEEK 2 PREVIEW

**Starting Monday, Feb 3** (same pattern):
- 4 new operations: SCAL, DSCAL, CSCAL, ZDSCAL (scaling)
- Same team, same process
- Expected to be faster (team now knows the pattern)
- Still same timeline: Kickoff Mon, code Tue-Thu, retro Fri

**Weeks 2-12 pace:** 50 operations/week (scaling mode)

---

## 📞 QUICK CONTACTS

- **Tech Lead:** [In standup or Slack]
- **QA Questions:** [Ask in standup]
- **Build Issues:** [Tell TL immediately]
- **Code Review:** Submit to TL repo (Tue-Thu)

---

**🎯 REMEMBER: Clear requirements + Daily standup + Code review = SUCCESS**

**Let's launch Week 1! 🚀**

---

*This quick reference: 2-minute read, 5-day resource*  
*Bookmark or print for access during coding*  
*Full details: See WEEK1_TEAM_KICKOFF_GUIDE.md*
