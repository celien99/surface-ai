# Task 4 Report: KnowledgeEvolution (changelog CRUD)

**Status:** DONE_WITH_CONCERNS

**Commit:** `caaa910` — `feat(knowledge): ✨ 添加 KnowledgeEvolution 变更日志（Append/GetHistory/GetChangesSince）`

**Tests:** 336 total (333 prior + 3 new), 0 failures.

**New tests (3):**
| Test | What it verifies |
|------|-----------------|
| `KnowledgeEvolutionTest.AppendAndGetHistory` | Append returns void, GetHistory returns entry with correct entity_type/entity_id/operation/version |
| `KnowledgeEvolutionTest.VersionIncrements` | Three Appends on same entity get versions 1, 2, 3 |
| `KnowledgeEvolutionTest.GetChangesSince` | Second entry (Node 2) returned when querying after first append |

**Files created:**
- `include/sai/knowledge/knowledge_evolution.h` — EvolutionOp enum, EvolutionEntry struct, KnowledgeEvolution class with Append/GetHistory/GetChangesSince

**Files modified:**
- `src/knowledge/knowledge_evolution.cpp` — full implementation (previously 0-byte stub)
- `tests/knowledge/knowledge_evolution_test.cpp` — 3 test cases (previously 0-byte stub)

**Concerns:**

1. **Brief's `GetChangesSince` timing issue (fixed in implementation).** The brief's reference code stored timestamps with second precision (`datetime('now')`) and compared with `>`. When two Appends and `after_first` all fall within the same second, the SQL comparison `WHERE timestamp > ?` cannot distinguish them, causing `GetChangesSince` to return 0 entries. **Fix:** Append now computes timestamps in C++ with microsecond precision (`YYYY-MM-DD HH:MM:SS.uuuuuu`) and binds them as parameters. `GetChangesSince` similarly formats the `since` parameter with microsecond precision. This makes lexicographic SQL string comparison reliably distinguish entries at microsecond granularity.

2. **Brief's anonymous namespace declarations would cause linker error (fixed).** The brief forward-declared `RecordToJson` and `JsonToRecord` inside the anonymous namespace (internal linkage), shadowing the same-named functions from `knowledge_record.h`. The callers in the same translation unit would resolve to the anonymous namespace declarations, which have no definitions, causing an undefined symbol link error. **Fix:** Removed these two forward declarations from the anonymous namespace. The functions are already declared in `knowledge_record.h` within the same `sai::knowledge` namespace and are accessible directly.

3. **Unused includes (fixed).** Removed `<source_location>` (unused in this translation unit). Added `<cstring>` for `std::strcmp` and `<ctime>` for `std::gmtime`/`std::mktime`, both used but omitted from the brief's reference code.

4. **Five `[[nodiscard]]` warnings in test file (not fixed, matches existing pattern).** The test file ignores return values of `Append()` in `VersionIncrements` and `GetChangesSince` tests (5 sites). This follows the same pattern as the existing `knowledge_graph_test.cpp` which also discards return values for non-critical intermediate calls. These are warnings, not errors, and are consistent with project conventions.
