#include "benchmark.h"

// Init profiles
struct bm_profile_t bm_profiles[BM_MAX_PROFILES] = {0};
// Initially, there's only one, the default profile
unsigned int bm_num_profiles = 1;
// We set the default profile as the initial profile
struct bm_profile_t *bm_current_profile = &bm_profiles[0];

// Initially we don't track any run yet, hence there's no run on the stack yet
__thread int bm_num_open_runs = 0;
__thread int bm_num_unopened_runs = 0;
__thread struct bm_run_t *open_runs_stack[BM_MAX_CONCURRENT_OPEN_PROFILES];

struct bm_summary_column_t bm_summary[64] = {0};
int bm_summary_counter = 0;


