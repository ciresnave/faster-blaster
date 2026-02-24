# Week 1 Team Kickoff Guide
**Phase 2 Implementation - Week 1**  
**January 27, 2026 (Monday)**  
**Duration: 4 hours (1:00 PM - 5:00 PM)**

---

## 📋 Agenda

### 1:00 PM - 1:45 PM: Resource Review (45 minutes)

**Assigned to: All team members**

**Actions:**
- [ ] Open and review: **PHASE2_START_GUIDE.md** (7-step execution)
- [ ] Open and review: **MULTI_PRECISION_VARIANTS_GUIDE.md** (implementation patterns)
- [ ] Open and review: **DEVELOPER_QUICK_REFERENCE.md** (commands & tools)
- [ ] Open and review: **WEEK1_DETAILED_EXECUTION_PLAN.md** (your specific assignments)
- [ ] Open: **MASTER_IMPLEMENTATION_SPREADSHEET_TIER1.csv** (tracking)

**Key Documents Location:**
```
c:\Users\cires\OneDrive\Documents\projects\faster-blaster\
├── PHASE2_START_GUIDE.md
├── MULTI_PRECISION_VARIANTS_GUIDE.md
├── DEVELOPER_QUICK_REFERENCE.md
├── WEEK1_DETAILED_EXECUTION_PLAN.md
├── WEEK1_TEAM_KICKOFF_GUIDE.md (THIS FILE)
├── WEEK1_STATUS_DASHBOARD.ps1
└── MASTER_IMPLEMENTATION_SPREADSHEET_TIER1.csv
```

**Questions to Answer While Reading:**
1. What is the overall project goal? → **603 Tier 1 operations in 12 weeks**
2. What are we implementing this week? → **SAXPY, DAXPY, CAXPY, ZAXPY (4 operations)**
3. What do I personally need to do? → **Check your assignment in WEEK1_DETAILED_EXECUTION_PLAN.md**
4. How do I test my implementation? → **Use test templates in tests/ directory**
5. How do I report progress? → **Update tracking spreadsheet daily**

**Time Breakdown:**
- 15 min: Quick read of PHASE2_START_GUIDE
- 15 min: Multi-precision pattern (your implementation template)
- 10 min: Quick reference (bookmark this!)
- 5 min: Your personal assignments

---

### 1:45 PM - 2:30 PM: Technical Verification (45 minutes)

**Technical Lead (TL):**
- [ ] Verify build system works:
  ```powershell
  cd c:\Users\cires\OneDrive\Documents\projects\faster-blaster
  cd build
  cmake --build . --config Release -j4
  ctest --verbose
  ```
- [ ] Confirm output shows: **"All tests passed"** or **"Test project: complete"**
- [ ] Report result to team: ✓ Pass/✗ Fail

**QA Engineer:**
- [ ] Set up test environment:
  - [ ] Locate test directory: `tests/` in project root
  - [ ] Verify test file exists: `test_blas_l1_saxpy.c` (CREATED FRESH TODAY)
  - [ ] Install Valgrind (if not installed):
    ```powershell
    # Windows: Use WSL or Docker, or Valgrind in MinGW
    valgrind --version
    ```
  - [ ] Prepare test execution script
  - [ ] Report readiness: ✓ Ready / ✗ Blockers

**Developers (Dev1, Dev2, Dev3):**
- [ ] Set up IDE/editor:
  - [ ] Open `c:\Users\cires\OneDrive\Documents\projects\faster-blaster\`
  - [ ] Set working directory to: `faster-blaster-reference/src/blas/level1/`
  - [ ] Verify files exist:
    ```
    - saxpy.c (existing reference - 44 lines)
    - daxpy.c (TO CREATE)
    - caxpy.c (TO CREATE)
    - zaxpy.c (TO CREATE)
    ```
  - [ ] Open `MULTI_PRECISION_VARIANTS_GUIDE.md` as reference
  - [ ] Report readiness: ✓ Ready / ✗ Blockers

---

### 2:30 PM - 3:15 PM: Team Standup (45 minutes)

**Moderator: Tech Lead**

**Format: 5-minute round-robin + 15 minutes discussion**

#### Round-Robin (Each person: ~4 minutes):

**Tech Lead:**
- Status: ✓ Build verified
- Blockers: None identified
- Plan for Tues-Thurs: Code reviews, architecture guidance
- Questions? None at kickoff

**QA Engineer:**
- Status: ✓ Test environment ready
- Blockers: (Report any)
- Plan: Create 32+ tests (8 per operation)
- Questions: Confirm test framework is Unity?

**Developer 1 (SAXPY/DAXPY):**
- Status: ✓ IDE set up
- Blockers: (Report any)
- Plan: Verify SAXPY → Implement DAXPY → Test both
- Questions: Confirm stride handling from reference?

**Developer 2 (CAXPY):**
- Status: ✓ IDE set up
- Blockers: (Report any)
- Plan: Implement CAXPY (complex float handling)
- Questions: How to handle C23 _Complex type?

**Developer 3 (ZAXPY):**
- Status: ✓ IDE set up
- Blockers: (Report any)
- Plan: Implement ZAXPY (complex double)
- Questions: Same as Dev2 for _Complex handling?

#### Team Discussion (15 minutes):

**Topic 1: Stride Handling (5 min)**
- Reference saxpy.c shows stride optimization pattern
- Pattern: `int ix = (incx > 0) ? 0 : (n-1)*(-incx)`
- All 4 operations use EXACT same pattern
- Questions: Any concerns with this approach?

**Topic 2: Complex Type Handling (5 min)**
- C23 provides `_Complex` for C and Z variants
- floatComplex = float _Complex (or float complex with C99)
- doubleComplex = double _Complex
- Naming: caxpy_ref() and zaxpy_ref()
- Pattern: `y[iy] += alpha * x[ix]` (C handles complex multiplication automatically)
- Questions: Everyone comfortable with C23 complex?

**Topic 3: Testing Strategy (5 min)**
- 8 tests per operation (32 total)
- Test template provided: `test_blas_l1_saxpy.c` (created today)
- Tests cover: basic, stride, n=0, alpha=0, n=1, negative alpha, backward, large values
- Replicate for DAXPY, CAXPY, ZAXPY with appropriate precision
- Questions: Any blockers on testing approach?

---

### 3:15 PM - 3:45 PM: Individual Assignment Review (30 minutes)

**Read Individually: Your Role Assignment**

Each person reviews their specific assignments from WEEK1_DETAILED_EXECUTION_PLAN.md:

#### Developer 1:
- **Monday (3:15-5:00 PM)**: Prepare SAXPY test harness (add #include guards, test struct)
- **Tuesday 9:00-10:00**: Verify SAXPY (add comments, check edge cases)
- **Tuesday 10:15-11:30**: DAXPY implementation (copy saxpy.c, replace `float` with `double`, adjust tolerance)
- **Tuesday 12:30-14:00**: Test DAXPY (run test suite, fix issues)
- **Tuesday 15:00-16:00**: Code review prep (document changes)

#### Developer 2:
- **Monday (3:15-5:00 PM)**: Study C23 _Complex type basics
- **Tuesday 9:00-11:30**: CAXPY implementation
  - Include: `#include <complex.h>`
  - Replace types: `typedef float _Complex floatComplex`
  - Replace tolerance: 1e-6 per component (magnitude check)
- **Tuesday 12:30-15:00**: CAXPY testing

#### Developer 3:
- **Monday (3:15-5:00 PM)**: Study C23 _Complex for double
- **Tuesday 9:00-11:30**: ZAXPY implementation
  - Include: `#include <complex.h>`
  - Replace types: `typedef double _Complex doubleComplex`
  - Replace tolerance: 1e-15 per component
- **Tuesday 12:30-15:00**: ZAXPY testing

#### QA Engineer:
- **Monday (3:15-5:00 PM)**: Template 8-test pattern for each operation
- **Tuesday morning**: Implement tests for SAXPY
- **Tuesday afternoon**: Run tests, document pass/fail

#### Tech Lead:
- **Monday (3:15-5:00 PM)**: Prepare code review checklist
- **Tuesday onwards**: 1-hour reviews per operation (4 operations = 4 hours total)

---

### 3:45 PM - 4:00 PM: Clarifying Questions (15 minutes)

**Open Floor for Questions**

Questions to expect/address:

**Q1: "Where is the reference implementation?"**  
A: `faster-blaster-reference/src/blas/level1/saxpy.c` (44 lines, full implementation)

**Q2: "What happens if I need help?"**  
A: Slack/Teams channel + daily 10 AM standup. Tech Lead available for architecture questions.

**Q3: "How do I handle negative strides?"**  
A: Reference code shows pattern. Both incx/incy can be negative. Test with backward indexing.

**Q4: "What about thread safety?"**  
A: BLAS operations are thread-safe if each thread uses different arrays. No internal state.

**Q5: "Can I start on Tuesday implementations today?"**  
A: Yes, but prioritize: (1) Understand SAXPY pattern, (2) Prepare test structure, (3) Then implement.

**Q6: "What's the Valgrind command?"**  
A: `valgrind --leak-check=full --show-leak-kinds=all ./test_blas_l1_saxpy`

---

### 4:00 PM - 4:45 PM: Build Success Verification (45 minutes)

**Entire Team + TL:**

1. **All developers**: Open SAXPY file
   - Location: `faster-blaster-reference/src/blas/level1/saxpy.c`
   - Verify: File exists and has 44 lines
   - Action: Read through code once (5 minutes) - understand stride logic

2. **All devs + QA**: Open test file
   - Location: `tests/test_blas_l1_saxpy.c`
   - Verify: File created fresh today with 8 test functions
   - Action: Identify test patterns you'll replicate (5 minutes)

3. **TL reports**: Final build status
   - Build completed: ✓ Yes / ✗ No
   - All systems operational: ✓ Yes / ✗ No
   - Team ready to execute: ✓ Yes / ✗ No

4. **Update tracking**:
   - Open: MASTER_IMPLEMENTATION_SPREADSHEET_TIER1.csv
   - Set: "kickoff_status" = "COMPLETE" for today
   - Set: "team_readiness" = "95%" (minor tweaks possible Tues morning)

---

### 4:45 PM - 5:00 PM: Wrap-up & Tomorrow's Prep (15 minutes)

**Tech Lead Summarizes:**

**What We Accomplished Today:**
- ✅ All team members understand project scope
- ✅ Build system verified
- ✅ Test environment ready
- ✅ Reference implementation located
- ✅ Roles and assignments clear
- ✅ First week targets confirmed (4 ops, 32 tests, Friday finish)

**What Happens Tuesday:**
- 9:00 AM: Team standup (10 min)
- 9:15 AM: Dev 1 begins SAXPY verification + DAXPY implementation
- 9:15 AM: Dev 2 begins CAXPY work
- 9:15 AM: Dev 3 begins ZAXPY work
- 9:15 AM: QA begins creating test suite
- 9:15 AM: TL begins code reviews

**Success Criteria for Friday 5 PM:**
- [ ] 4 operations implemented (SAXPY, DAXPY, CAXPY, ZAXPY)
- [ ] 32+ tests created and passing
- [ ] Valgrind: Clean (zero leaks, zero errors)
- [ ] Code reviews: All approved (0 critical issues)
- [ ] Tracking: 100% updated
- [ ] Ready for Week 2 (SCAL family)

**Homework (Optional, but Recommended):**

For all developers:
- Quick read of multi-precision pattern (15 minutes)
- Think about complex number handling (5 minutes)

For Dev 2 & Dev 3:
- Watch 5-minute video on C23 _Complex: https://en.cppreference.com/w/c/numeric/complex
- Or read: https://en.wikibooks.org/wiki/C_Programming/stdint.h

For QA:
- Prepare 8-test template structure (write it in a scratch doc)
- Think about edge cases to test

---

## 📊 Post-Meeting Checklist

**After 5 PM today, each person should:**

- [ ] Email tech lead: "Kickoff complete, ready for Tuesday"
- [ ] Update tracking spreadsheet with "kickoff_status = COMPLETE"
- [ ] Close all resource documents (keep bookmarks!)
- [ ] Review your personal assignments once more
- [ ] Prepare any questions for first standup Tuesday 9 AM
- [ ] (Optional) Start reading multi-precision guide if time permits

---

## 🔗 Quick Links (Bookmark These)

**Essential Resources:**
1. **Start Guide**: PHASE2_START_GUIDE.md
2. **Implementation Pattern**: MULTI_PRECISION_VARIANTS_GUIDE.md
3. **Quick Reference**: DEVELOPER_QUICK_REFERENCE.md
4. **Your Assignments**: WEEK1_DETAILED_EXECUTION_PLAN.md
5. **Tracking Progress**: MASTER_IMPLEMENTATION_SPREADSHEET_TIER1.csv
6. **Today's Status**: WEEK1_STATUS_DASHBOARD.ps1

**Source Code:**
- Reference SAXPY: `faster-blaster-reference/src/blas/level1/saxpy.c`
- Test Template: `tests/test_blas_l1_saxpy.c`
- BLAS Header: `faster-blaster-reference/include/blas_reference.h`

**Build Commands:**
```powershell
# Build
cd build
cmake --build . --config Release -j4

# Test
ctest --verbose

# Memory check
valgrind ./test_blas_l1_saxpy
```

---

## 🎯 Final Reminder

**We Are Starting a 24-Week Implementation Plan**

- **Week 1 Goal**: Validate workflow with 4 operations
- **Weeks 2-12**: Scale to 50 ops/week (12 weeks × 50 = 600 ops)
- **Weeks 13-24**: Implement advanced features + LAPACK/sparse ops

**This Week is Critical**: If we succeed with SAXPY/DAXPY/CAXPY/ZAXPY, we know the process works and can scale.

**Team Mantra**: *Clear requirements + Daily standup + Code review = Success*

---

## 📧 Contact

**Tech Lead**: [Primary contact for architecture questions]  
**QA Lead**: [Test framework and validation questions]  
**Development PM**: [Blockers, resource requests, timeline issues]

**Daily Standup**: Tuesday-Friday, 9:00-9:15 AM  
**Status Dashboard**: Run `WEEK1_STATUS_DASHBOARD.ps1` daily for quick overview

---

**KICKOFF COMPLETE - Team ready to execute Tuesday, January 28, 2026 at 9:00 AM**

*Next file to create: Daily standup notes template (WEEK1_DAILY_STANDUP_TEMPLATE.md)*
