#include <algorithm>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

// Pure integer checks. No Python objects, database access or search mutation.
extern "C" int restricted_children(
    const int64_t* roots, int64_t depth, const int64_t* parents, int64_t count,
    const uint8_t* blocked, const int64_t* clause_offsets, int64_t clauses,
    const int64_t* indices, const int64_t* variant_offsets,
    const int64_t* variants, const int64_t* variant_counts,
    int threads, uint8_t* admitted) noexcept {
  if (depth < 0 || count < 0 || threads < 1) return -1;
  std::vector<std::thread> pool;
  try {
    int workers = std::min<int64_t>(threads, count);
    auto work = [&](int worker) {
      for (int64_t k = worker; k < count; k += workers) {
        int64_t root = parents[k] == -1 ? depth : roots[parents[k]];
        bool ok = !blocked[root];
        for (int64_t c = 0; ok && c < clauses; ++c) {
          int64_t start = clause_offsets[c], length = clause_offsets[c+1]-start;
          bool match = false;
          for (int64_t v = 0; !match && v < variant_counts[c]; ++v) {
            match = true;
            for (int64_t j = 0; j < length; ++j) {
              int64_t index = indices[start+j];
              if (index > depth) break;
              int64_t value = index == depth ? root : roots[index];
              if (variants[variant_offsets[c]+v*length+j] != value) {
                match = false; break;
              }
            }
          }
          ok = match;
        }
        admitted[k] = ok;
      }
    };
    for (int w = 1; w < workers; ++w) pool.emplace_back(work, w);
    if (workers) work(0);
    for (auto& t : pool) t.join();
    return workers;
  } catch (...) {
    for (auto& t : pool) if (t.joinable()) t.join();
    return -2;
  }
}
