#ifndef BENCHMARK_CUH_
#define BENCHMARK_CUH_

#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <errno.h>

#include "get_real_time.h"

/**
 * General concepts:
 *
 * We have profiles, metrics, runs and data points.
 * Think of each profile as a table (such as in a spreadsheet), with metrics being the table's column headings (e.g., "sort duration", "merge duration").
 * Each row represents a run along with the data points that have been captured for that specific run (row) and a certain metric (column).
 *
 * Capturing
 * A profile has an identifier, such that one can easily select the profile for subsequent runs.
 * That is, as long as we're not capturing a run, we can switch to a different profile.
 * If no profile is selected, we use the default profile, which is identified ....
 *
 * Nested runs:
 *
 * Other remarks:
 *
 * As a new run is being captured there is a small overhead associated when the run is set up.
 * A more negligible overhead is associated when metrics are set for a run.
 */

/****************************
 * DEBUGGING
 ****************************/
#define BM_DEBUG_TRACE 0
#if BM_DEBUG_TRACE
	#define BM_DBG_TRACE(msg, ...)  printf(msg "\n", ##__VA_ARGS__);
#else
	#define BM_DBG_TRACE(msg, ...)
#endif

// Determines whether we allow capturing a run while we're already capturing another run
// If this is false, we assign the nested runs' data points to the parent (root) run.
#define BM_ALLOW_NESTED_RUNS 0

// Whenever a new metric is added, we search for possible duplicates and merge if available
// Introduces slight overhead in O(n), where n denotes the number of the profile's metrics.
#define BM_MERGE_DUPLICATES_ON_ADD 0

// The maximum number of profiles being supported
#define BM_MAX_PROFILES 32

// The maximum string length supported for profile identifiers
#define BM_MAX_PROFILE_ID_LEN 64

// The maximum string length supported for run identifiers
#define BM_MAX_RUN_ID_LEN 64

// The maximum string length supported for a metric's name
#define BM_MAX_METRIC_LEN 64

// The maximum number of metrics supported for a single profile
#define BM_MAX_METRICS 64

// Number of runs that may simultaneously be opened
#define BM_MAX_CONCURRENT_OPEN_PROFILES 64


enum bm_metric_type
{
	BM_METRIC_T_UNDEF = 0,		// undefined type
	BM_METRIC_T_UINT,			// 32-bit uint
	BM_METRIC_T_HEX,			// 32-bit uint, print as as hex
	BM_METRIC_T_INT,			// 32-bit int
	BM_METRIC_T_DOUBLE,			// double precision floating point
	BM_METRIC_T_CPU_EVT,		// elapsed time based on CPU wall-clock time
#ifdef __CUDACC__
	BM_METRIC_T_CU_EVT,			// elapsed time based on GPU wall-clock time
#endif
};


/******************
 * METRICS        *
 ******************/
#ifdef __CUDACC__
struct bm_cu_evt_pair_t
{
	cudaEvent_t cu_evt_start;
	cudaEvent_t cu_evt_stop;
};
#endif

struct bm_cpu_time_pair_t
{
	double cpu_start;
	double cpu_end;
};

struct bm_data_point_t {
	enum bm_metric_type m_type;
	union {
		unsigned int u;
		unsigned long long int llu;
		int i;
		long long int lli;
		double f;
		struct bm_cpu_time_pair_t cpu_evt;
#ifdef __CUDACC__
		struct bm_cu_evt_pair_t cu_evt;
#endif
	} value;
};

struct bm_run_t {
	char run_name[BM_MAX_RUN_ID_LEN];
	struct bm_data_point_t data_points[BM_MAX_METRICS];
};

struct bm_profile_t {
	uint_fast32_t 		num_runs;		// number of runs opened
	uint_fast32_t 		num_alloc;		// number of runs that have been allocated
	struct bm_run_t 	**runs;				// memory pointers to the actual runs

	uint_fast32_t num_metrics;
	char metrics[BM_MAX_METRICS][BM_MAX_METRIC_LEN];
	char id[BM_MAX_PROFILE_ID_LEN];
};

struct bm_summary_column_t
{
	char caption[128];
	double value;
};

extern struct bm_profile_t bm_profiles[BM_MAX_PROFILES];
extern unsigned int bm_num_profiles;
extern struct bm_profile_t *bm_current_profile;
extern struct bm_summary_column_t bm_summary[64];
extern int bm_summary_counter;

extern __thread int bm_num_open_runs;
extern __thread int bm_num_unopened_runs;
extern __thread struct bm_run_t *open_runs_stack[BM_MAX_CONCURRENT_OPEN_PROFILES];

inline struct bm_profile_t *bm_get_or_add_profile(const char *profile_id)
{
	if(profile_id){
		for(int i=0;i<bm_num_profiles;i++){
			if(strcmp(profile_id, bm_profiles[i].id) == 0){
				return &bm_profiles[i];
			}
		}
		bm_profiles[bm_num_profiles] = (struct bm_profile_t){0};
		strcpy(bm_profiles[bm_num_profiles].id, profile_id);
		bm_num_profiles++;
		return &bm_profiles[bm_num_profiles];
	}else {
		return &bm_profiles[0];
	}
}

/**
 * Pushes a new run onto the given environment's stack
 */
inline struct bm_run_t **bm_env_push_new_run(struct bm_profile_t *bm_profile, struct bm_run_t *new_run, float cfg_alloc_grow, uint_fast32_t cfg_alloc_min)
{
	BM_DBG_TRACE("Getting new item. \n -> Item count / allocated: %3" PRIuFAST32 ", %3" PRIuFAST32 "\n", bm_current_profile->num_runs, bm_current_profile->num_alloc);
	if(bm_profile->num_runs == bm_profile->num_alloc){
		float grow = cfg_alloc_grow > 1 ? cfg_alloc_grow : 1.5f;
		uint_fast32_t alloc_min = cfg_alloc_min > 1 ? cfg_alloc_min : 1;
		bm_profile->num_alloc *= grow;
		if(bm_profile->num_alloc < alloc_min)
			bm_profile->num_alloc = alloc_min;
		BM_DBG_TRACE(" --> Allocating for: %3" PRIuFAST32 "\n", bm_current_profile->num_alloc);
		bm_profile->runs = (struct bm_run_t **)realloc(bm_profile->runs, bm_profile->num_alloc * sizeof(*bm_profile->runs));
	}
	bm_profile->runs[bm_profile->num_runs] = new_run;
	bm_profile->num_runs++;
	return &bm_profile->runs[bm_profile->num_runs];
}

inline void bm_set_profile(const char *profile_id)
{
	// Only if this is no active run we allow setting the profile
	if(bm_num_open_runs == 0){
		// Get the existing profile with the given id, if there is one.
		// Otherwise, simply add a new profile with the given id.
		bm_current_profile = bm_get_or_add_profile(profile_id);
	}
}

/**
 * Starts capturing a new run in the given profile_id.
 * @param profile_id The profile id, this run should be captured for.
 * @return returns true if a new run is being captured and false if not.
 */
inline bool bm_capture_new_run_in_profile(const char *run_id, const char *profile_id)
{
	if(BM_ALLOW_NESTED_RUNS || bm_num_open_runs == 0){
		// Set the current profile
		bm_set_profile(profile_id);

		// Allocate memory for a new run
		struct bm_run_t *new_run_ptr = (struct bm_run_t *) malloc(sizeof(struct bm_run_t));

		// Push the run onto the current benchmark environment's stack
		bm_env_push_new_run(bm_current_profile, new_run_ptr, 1.5, 512);
		open_runs_stack[bm_num_open_runs] = new_run_ptr;
		BM_DBG_TRACE(" -> Open new run #%d (%p), currently open: %d ", bm_current_profile->num_runs, open_runs_stack[bm_num_open_runs], bm_num_open_runs);
		memset(open_runs_stack[bm_num_open_runs]->data_points, 0, sizeof(open_runs_stack[bm_num_open_runs]->data_points));
		if(run_id)
			strcpy(open_runs_stack[bm_num_open_runs]->run_name, run_id);
		else
			strcpy(open_runs_stack[bm_num_open_runs]->run_name, "UNDEF");
		bm_num_open_runs++;
		return true;
	}else{
		bm_num_unopened_runs++;
		return false;
	}
}

/**
 * Starts capturing a new run in the default profile.
 */
inline bool bm_capture_new_run(const char *run_id)
{
	return bm_capture_new_run_in_profile(run_id, NULL);
}

/**
 * Removes the thread's currently active run.
 */
inline void bm_close_run()
{
	BM_DBG_TRACE(" -> Closing current run");
	if(bm_num_unopened_runs>0)
		bm_num_unopened_runs--;
	else
		if(bm_num_open_runs>0)
			bm_num_open_runs--;
}

/**
 * Returns the thread's current run. If there's no open run, it'll create a new one and return that instead.
 */
inline struct bm_run_t *bm_get_current_run()
{
	if(bm_num_open_runs == 0 || open_runs_stack[bm_num_open_runs-1] == NULL)
		bm_capture_new_run(NULL);
	BM_DBG_TRACE(" -> Get current run %p", open_runs_stack[bm_num_open_runs-1]);
	return open_runs_stack[bm_num_open_runs-1];
}

/**
 * Adds a new metric to the benchmark environment.
 * @param metric_caption The metrics caption
 * @return The metric index under which it was added to the benchmark environment
 */
inline int bm_get_or_add_metric(const char *metric_caption)
{
	if(BM_MERGE_DUPLICATES_ON_ADD){
		for(int m=0; m<bm_current_profile->num_metrics; m++){
			if(strcmp(bm_current_profile->metrics[m], metric_caption) == 0){
				return m;
			}
		}
	}
	int metric_idx = bm_current_profile->num_metrics++;
	strcpy(bm_current_profile->metrics[metric_idx], metric_caption);
	return metric_idx;
}

inline int bm_get_or_add_metric_array_index(const char *metric_caption, int index)
{
	char buff[128];
	sprintf(buff, "%s[%d]", metric_caption, index);
	for(int m=0; m<bm_current_profile->num_metrics; m++){
		if(strcmp(bm_current_profile->metrics[m], buff) == 0)
			return m;
	}
	int metric_idx = bm_current_profile->num_metrics++;
	strcpy(bm_current_profile->metrics[metric_idx], buff);
	return metric_idx;
}

/**
 * Helps retrieving a pointer to the metric of the benchmark environment at the given metric index for the thread's current run.
 * @param metric_idx
 * @return
 */
inline struct bm_data_point_t *bm_get_current_run_metric_ptr(const int metric_idx)
{
	struct bm_run_t *current_run = bm_get_current_run();
	return &current_run->data_points[metric_idx];
}

inline void bm_print_metric(struct bm_data_point_t *metric, char buffer[])
{
	switch (metric->m_type) {
		case BM_METRIC_T_UNDEF:		sprintf(buffer, "%s",""); 					break;
		case BM_METRIC_T_UINT: 		sprintf(buffer, "%u", metric->value.u); 	break;
		case BM_METRIC_T_HEX: 		sprintf(buffer, "0x%08x", metric->value.u); break;
		case BM_METRIC_T_INT: 		sprintf(buffer, "%d", metric->value.i); 	break;
		case BM_METRIC_T_DOUBLE: 	sprintf(buffer, "%.3f", metric->value.f); 	break;
#ifdef __CUDACC__
		case BM_METRIC_T_CU_EVT:
			float elapsed_time;
			cudaEventElapsedTime(&elapsed_time, metric->value.cu_evt.cu_evt_start, metric->value.cu_evt.cu_evt_stop);
			sprintf(buffer, "%.3f", elapsed_time);
			break;
#endif
		case BM_METRIC_T_CPU_EVT:
			sprintf(buffer, "%.3f", (metric->value.cpu_evt.cpu_end-metric->value.cpu_evt.cpu_start)*1000);
			break;
		default:
			break;
	}
}


/**
 * Merges metrics that share the same metric name, re-mapping all those data points of duplicate metrics of a higher index to the metrics with a lower index.
 */
inline void bm_merge_metrics(struct bm_profile_t *profile)
{
	// We iterate over all metrics (potential candidate of being a duplicate of another metric)
	for(int dup_candidate_idx=1; dup_candidate_idx<profile->num_metrics; dup_candidate_idx++){

		// We try to find the index of a metric sharing the same as our candidate
		for(int original_metric_idx=0; original_metric_idx<dup_candidate_idx; original_metric_idx++){

			// If the metric names match we have a duplicate at index dup_candidate_idx, so we remap to the original metric original_metric_idx
			if(strcmp(profile->metrics[original_metric_idx], profile->metrics[dup_candidate_idx]) == 0){

				// Copy data points from this metric to the original metric
				for(int run=0;run<profile->num_runs;run++){
					if(profile->runs[run]->data_points[original_metric_idx].m_type == BM_METRIC_T_UNDEF)
						profile->runs[run]->data_points[original_metric_idx] = profile->runs[run]->data_points[dup_candidate_idx];
				}

				// Shift every metric from a higher index one down
				for(int run=0;run<profile->num_runs;run++){
					for (int metric_idx = dup_candidate_idx; metric_idx < profile->num_metrics-1; ++metric_idx) {
							profile->runs[run]->data_points[metric_idx] = profile->runs[run]->data_points[metric_idx+1];
					}
				}
				for (int metric_idx = dup_candidate_idx; metric_idx < profile->num_metrics-1; ++metric_idx)
					strcpy(profile->metrics[metric_idx], profile->metrics[metric_idx+1]);

				// Continue search at the current index (with next iteration), as we moved everything to one index below
				dup_candidate_idx--;

				// We found a duplicate metric, so basically we have one metric less
				profile->num_metrics--;

				// Exit our search for the original metric
				break;
			}
		}
	}
}

inline void bm_write_environment_runs(FILE *fp, int min_col_width, int max_col_width, char comma_delimiter, const char *enclosing_quotes_char, const char *run_name)
{
	bm_merge_metrics(bm_current_profile);
#ifdef __CUDACC__
	// Make sure we've got all cuda events from the benchmarks
	cudaDeviceSynchronize();
#endif

	char buff[256];
	char tmp_buff[256];
	int filtered_metric_ids[BM_MAX_METRICS];
	int metric_str_len[BM_MAX_METRICS] = {0};
	for(int i=0;i<BM_MAX_METRICS; i++){
		filtered_metric_ids[i] = 0;
	}

	int max_run_name_len = 0;
	for(int p=0; p<bm_current_profile->num_runs; p++){
		if(run_name  == NULL || strcmp(bm_current_profile->runs[p]->run_name, run_name) == 0){
			// Determine the length of the longest run name (first column)
			if(max_run_name_len < strlen(bm_current_profile->runs[p]->run_name))
				max_run_name_len = strlen(bm_current_profile->runs[p]->run_name);

			// Determine the metrics to include and the length of the longest metric value
			for(int m=0; m<bm_current_profile->num_metrics; m++){
				if(bm_current_profile->runs[p]->data_points[m].m_type != BM_METRIC_T_UNDEF){
					filtered_metric_ids[m] = 1;

					bm_print_metric(&bm_current_profile->runs[p]->data_points[m], tmp_buff);
					sprintf(buff, "%*.*s", min_col_width, max_col_width, tmp_buff);
					unsigned int str_len = strlen(buff);
					metric_str_len[m] = str_len > metric_str_len[m] ? str_len : metric_str_len[m];
				}
			}
		}
	}
	for(int m=0; m<bm_current_profile->num_metrics; m++){
		if(filtered_metric_ids[m]){
			unsigned int str_len = strlen(bm_current_profile->metrics[m]);
			metric_str_len[m] = str_len > metric_str_len[m] ? str_len : metric_str_len[m];
		}
	}
	max_run_name_len += 2;

	// Print Captions
	fprintf(fp, "%s%*.*s%s%c", enclosing_quotes_char, max_run_name_len, max_run_name_len, "run", enclosing_quotes_char, comma_delimiter);
	for(int m=0; m<bm_current_profile->num_metrics; m++){
		if(filtered_metric_ids[m])
			fprintf(fp, "%s%*.*s%s%c", enclosing_quotes_char, metric_str_len[m], metric_str_len[m], bm_current_profile->metrics[m], enclosing_quotes_char, comma_delimiter);
	}
	fprintf(fp, "\n");

	// ITERATE THROUGH RUNS AND PRINT LINE BY LINE
	for(int p=0; p<bm_current_profile->num_runs; p++){
		if(run_name  == NULL || strcmp(bm_current_profile->runs[p]->run_name, run_name) == 0){
			// RUNS NAME
			fprintf(fp, "%s%*.*s%s%c", enclosing_quotes_char, max_run_name_len, max_run_name_len, bm_current_profile->runs[p]->run_name, enclosing_quotes_char, comma_delimiter);

			// Iterate over the metrics
			for(int m=0; m<bm_current_profile->num_metrics; m++){
				if(!filtered_metric_ids[m])
					continue;
				bm_print_metric(&bm_current_profile->runs[p]->data_points[m], buff);
				fprintf(fp, "%s%*.*s%s%c", enclosing_quotes_char, metric_str_len[m], metric_str_len[m], buff, enclosing_quotes_char, comma_delimiter);
			}
			fprintf(fp, "\n");
		}
	}
}

inline double bm_get_metric_numeric_value(struct bm_data_point_t *metric)
{
	switch (metric->m_type) {
		case BM_METRIC_T_UNDEF:		return 0.0;
		case BM_METRIC_T_UINT: 		return (double)metric->value.u;
		case BM_METRIC_T_HEX: 		return (double)metric->value.u;
		case BM_METRIC_T_INT: 		return (double)metric->value.i;
		case BM_METRIC_T_DOUBLE: 	return (double)metric->value.f;
#ifdef __CUDACC__
		case BM_METRIC_T_CU_EVT:
			float elapsed_time;
			cudaEventElapsedTime(&elapsed_time, metric->value.cu_evt.cu_evt_start, metric->value.cu_evt.cu_evt_stop);
			return (double) elapsed_time;
#endif
		case BM_METRIC_T_CPU_EVT:
			return (double)((metric->value.cpu_evt.cpu_end-metric->value.cpu_evt.cpu_start)*1000);
		default:
			return 0.0;
	}
}

inline double bm_get_run_metric_average(const char *run_name, const char *metric_name, double *min, double *max, double *avg)
{
	// Make sure we've got all cuda events from the benchmarks
	cudaDeviceSynchronize();

	double metric_sum = 0;
	double min_metric = 1000000000.0;
	double max_metric = -1000000000.0;
	double tmp;
	int metric_idx = bm_get_or_add_metric(metric_name);
	int counter = 0;
	for(int p=0; p<bm_current_profile->num_runs; p++){
		if(strcmp(bm_current_profile->runs[p]->run_name, run_name) == 0){
			tmp = bm_get_metric_numeric_value(&bm_current_profile->runs[p]->data_points[metric_idx]);
			if(tmp > max_metric)
				max_metric = tmp;
			if(tmp < min_metric)
				min_metric = tmp;
			metric_sum += tmp;
			counter++;
		}
	}

	if(min)
		*min = min_metric;
	if(max)
		*max = max_metric;
	if(avg)
		*avg = metric_sum/(double)counter;

	return metric_sum/(double)counter;
}

inline void bm_print_summary(FILE *fp, int min_col_width, int max_col_width, const char comma_delimiter, const char *enclosing_quotes_char)
{
	printf("%d summary columns\n", bm_summary_counter);
	for(int i=0; i < bm_summary_counter; i++){
		fprintf(fp, "%s%*.*s%s", enclosing_quotes_char, min_col_width, max_col_width, bm_summary[i].caption, enclosing_quotes_char);
		if(i<bm_summary_counter-1)
			fprintf(fp, "%c", comma_delimiter);
	}
	fprintf(fp, "\n");
	for(int i=0; i < bm_summary_counter; i++){
		fprintf(fp, "%s%*.*f%s", enclosing_quotes_char, min_col_width, 3, bm_summary[i].value, enclosing_quotes_char);
		if(i<bm_summary_counter-1)
			fprintf(fp, "%c", comma_delimiter);
	}
	fprintf(fp, "\n");
}

inline void bm_print_environment(const char *filter_by_run_name, const char delimeter='|')
{
	if(filter_by_run_name)
		bm_write_environment_runs(stdout, 14, 18, delimeter, "", filter_by_run_name);
	else
		bm_write_environment_runs(stdout, 14, 18, delimeter, "", NULL);
}

#define BM_SET_PROFILE(profile_id) \
bm_set_profile(profile_id);

// Open a new benchmark run. Subsequent metrics will be assigned to this run.
#define BM_OPEN_PROFILE(run_id, ...)										\
		char run_id_buff[BM_MAX_RUN_ID_LEN];								\
		sprintf(run_id_buff, "%s", run_id, ##__VA_ARGS__); 					\
		bm_capture_new_run(run_id_buff);									\


// Close the current benchmark run
#define BM_CLOSE_PROFILE()														\
	bm_close_run();

// Print all benchmark runs
#define BM_PRINT_ALL_PROFILES(delimeter)									\
	bm_print_environment(NULL, delimeter);

#define BM_PRINT_PROFILE(run_name)											\
	bm_print_environment(#run_name);

// Write out all benchmark runs
#define BM_LOG_ALL_PROFILES(out_path)											\
{																				\
	FILE *fp;																	\
	fp=fopen(#out_path, "w");													\
	if(!fp) printf("Error opening file %d\n", errno);							\
	printf("Writing benchmark log to %s\n", #out_path);							\
	bm_write_environment_runs(fp, 1, 100, ',', "\"", NULL);								\
	fclose(fp);																	\
}

#define BM_LOG_PROFILES(out_path, run_name) 								\
{																				\
	FILE *fp;																	\
	fp=fopen(#out_path, "w");													\
	if(!fp) printf("Error opening file %d\n", errno);							\
	printf("Writing benchmark log to %s\n", #out_path);							\
	bm_write_environment_runs(fp, 1, 100, ',', "\"", #run_name);		\
	fclose(fp);																	\
}

#define BM_PRINT_AND_WRITE_PROFILE(out_path, run_name, caption)				\
	BM_LOG_PROFILES(out_path, run_name)										\
	printf("\n-----------" caption "-----------\n");							\
	BM_PRINT_PROFILE(run_name);												\
	printf("-----------" caption "-----------\n\n");

#define BM_ADD_TO_SUMMARY(c_caption, run_name, avg_over_metric_name) 		\
{																				\
	struct bm_summary_column_t col;												\
	strcpy(col.caption, #c_caption);											\
	col.value = bm_get_run_metric_average(#run_name, #avg_over_metric_name, 0, 0, 0); \
	bm_summary[bm_summary_counter++] = col;\
}

#define BM_ADD_MINMAXAVG_TO_SUMMARY(c_caption, run_name, avg_over_metric_name) 		\
{																				\
	struct bm_summary_column_t col;												\
	struct bm_summary_column_t min;												\
	struct bm_summary_column_t max;												\
	strcpy(col.caption, #c_caption);											\
	strcpy(min.caption, #c_caption "min");										\
	strcpy(max.caption, #c_caption "max");										\
	bm_get_run_metric_average(#run_name, #avg_over_metric_name, &min.value, &max.value, &col.value); \
	bm_summary[bm_summary_counter++] = col;\
	bm_summary[bm_summary_counter++] = min;\
	bm_summary[bm_summary_counter++] = max;\
}

#define BM_RESET_SUMMARY() 														\
{																				\
	bm_summary_counter=0;														\
}

#define BM_LOG_SUMMARY(out_path)												\
{																				\
	FILE *fp;																	\
	fp=fopen(#out_path, "w");													\
	if(!fp) printf("Error opening file %d\n", errno);							\
	printf("Writing benchmark log to %s\n", #out_path);							\
	bm_print_summary(fp, 1, 100, ',', "\"");									\
	fclose(fp);																	\
}

#define BM_PRINT_SUMMARY()														\
	printf("\n----------- SUMMARY -----------\n");								\
	bm_print_summary(stdout, 18, 18, '|', "");									\
	printf("----------- SUMMARY -----------\n");

#define BM_PRINT_AND_LOG_SUMMARY(out_path) 										\
	BM_LOG_SUMMARY(out_path);													\
	BM_PRINT_SUMMARY()

// Declares the given metric in the current scope
#define BM_DECLARE_METRIC(caption, descriptor/*, BM_METRIC_T*/)					\
static int bm_metric_##descriptor##_idx = -1;								\
if(bm_metric_##descriptor##_idx == -1){										\
	bm_metric_##descriptor##_idx = bm_get_or_add_metric(#caption);			\
	/*bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx)->m_type = BM_METRIC_T; */\
}

// Assigns the given value as data point for the given metric, which has to be declared in this or parent scope
#define BM_ASSIGN_DATA_POINT(descriptor, rvalue,  BM_METRIC_T)				\
bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx)->value.u = rvalue;		\
bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx)->m_type = BM_METRIC_T;	\

// Declares and assigns the value to the metric
#define BM_DEFINE_DATA_POINT(caption, descriptor, rvalue,  BM_METRIC_T)			\
BM_DECLARE_METRIC(caption, descriptor)											\
BM_ASSIGN_DATA_POINT(descriptor, rvalue, BM_METRIC_T)

/****** SETTING METRICS *****/
#define BM_SET_METRIC_UINT(caption, descriptor, rvalue)							\
BM_DEFINE_DATA_POINT(caption, descriptor, rvalue, BM_METRIC_T_UINT)

#define BM_SET_METRIC_HEX4(caption, descriptor, rvalue)							\
BM_DEFINE_DATA_POINT(caption, descriptor, rvalue, BM_METRIC_T_HEX)

#define BM_SET_METRIC_INT(caption, descriptor, rvalue)							\
BM_DEFINE_DATA_POINT(caption, descriptor, rvalue, BM_METRIC_T_INT)

#define BM_SET_METRIC_DOUBLE(caption, descriptor, rvalue)						\
BM_DEFINE_DATA_POINT(caption, descriptor, rvalue, BM_METRIC_T_DOUBLE)


/****** CAPTURING PERFORMANCE METRICS *****/
#define BM_START_CPU_PROFILE(caption, descriptor)								\
BM_DECLARE_METRIC(caption, descriptor)										\
bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx)->m_type = BM_METRIC_T_CPU_EVT;	\
bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx)->value.cpu_evt.cpu_start = get_real_time();

#define BM_STOP_CPU_PROFILE(caption, descriptor);								\
bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx)->m_type = BM_METRIC_T_CPU_EVT;	\
bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx)->value.cpu_evt.cpu_end = get_real_time();

#ifdef __CUDACC__
#define BM_START_CUDA_EVENT(caption, descriptor, stream)						\
BM_DECLARE_METRIC(caption, descriptor)										\
bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx)->m_type = BM_METRIC_T_CU_EVT;								\
cudaEventCreate(&(bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx)->value.cu_evt.cu_evt_start));			\
cudaEventCreate(&(bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx)->value.cu_evt.cu_evt_stop));				\
cudaEventRecord((bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx)->value.cu_evt.cu_evt_start), stream);

#define BM_STOP_CUDA_EVENT(caption, descriptor, stream)																		\
	cudaEventRecord((bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx)->value.cu_evt.cu_evt_stop), stream);
#else
#define BM_START_CUDA_EVENT(caption, descriptor, stream)
#define BM_STOP_CUDA_EVENT(caption, descriptor, stream)
#endif



#define BM_DECLARE_METRIC_ARRAY(caption, descriptor, index, max_elements)		\
	static int bm_metric_##descriptor##_idx[max_elements];						\
	static char bm_metric_##descriptor##_flgs[max_elements] = {0};

#define BM_GET_METRIC_ARRAY_AT(caption, descriptor, index, max_elements)		\
	if(bm_metric_##descriptor##_flgs[index] == 0){								\
		bm_metric_##descriptor##_flgs[index] = 1;								\
		bm_metric_##descriptor##_idx[index] = bm_get_or_add_metric_array_index(#caption, index);	\
	}


/****** SETTING METRICS *****/
#define BM_SET_METRIC_ARRAY_UINT(caption, descriptor, rvalue, index, max_elements)							\
	BM_GET_METRIC_ARRAY_AT(caption, descriptor, index, max_elements)										\
	bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx[index])->m_type = BM_METRIC_T_UINT;	\
	bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx[index])->value.u = rvalue;

#define BM_SET_METRIC_ARRAY_HEX4(caption, descriptor, rvalue, index, max_elements)							\
	BM_GET_METRIC_ARRAY_AT(caption, descriptor, index, max_elements)										\
	bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx[index])->m_type = BM_METRIC_T_HEX;	\
	bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx[index])->value.u = rvalue;

#define BM_SET_METRIC_ARRAY_INT(caption, descriptor, rvalue, index, max_elements)							\
	BM_GET_METRIC_ARRAY_AT(caption, descriptor, index, max_elements)										\
	bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx[index])->m_type = BM_METRIC_T_INT;	\
	bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx[index])->value.i = rvalue;

#define BM_SET_METRIC_ARRAY_DOUBLE(caption, descriptor, rvalue, index, max_elements)						\
	BM_GET_METRIC_ARRAY_AT(caption, descriptor, index, max_elements)										\
	bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx[index])->m_type = BM_METRIC_T_DOUBLE;	\
	bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx[index])->value.f = rvalue;

/****** CAPTURING PERFORMANCE METRICS *****/
#define BM_START_CPU_ARRAY_PROFILE(caption, descriptor, index, max_elements)								\
	BM_GET_METRIC_ARRAY_AT(caption, descriptor, index, max_elements)										\
	bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx[index])->m_type = BM_METRIC_T_CPU_EVT;	\
	bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx[index])->value.cpu_evt.cpu_start = get_real_time();

#define BM_STOP_CPU_ARRAY_PROFILE(caption, descriptor, index, max_elements);								\
	bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx[index])->m_type = BM_METRIC_T_CPU_EVT;	\
	bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx[index])->value.cpu_evt.cpu_end = get_real_time();

#ifdef __CUDACC__
#define BM_START_CUDA_ARRAY_EVENT(caption, descriptor, stream, index, max_elements)						\
	BM_GET_METRIC_ARRAY_AT(caption, descriptor, index, max_elements)										\
	bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx[index])->m_type = BM_METRIC_T_CU_EVT;								\
	cudaEventCreate(&(bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx[index])->value.cu_evt.cu_evt_start));			\
	cudaEventCreate(&(bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx[index])->value.cu_evt.cu_evt_stop));				\
	cudaEventRecord((bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx[index])->value.cu_evt.cu_evt_start), stream);

#define BM_STOP_CUDA_ARRAY_EVENT(caption, descriptor, stream, index, max_elements)																		\
	cudaEventRecord((bm_get_current_run_metric_ptr(bm_metric_##descriptor##_idx[index])->value.cu_evt.cu_evt_stop), stream);

#define BM_START_CUDA_ARRAY_EVENT_LEV(lev, caption, descriptor, stream, index, max_elements)						\
	if(lev>1){ \
		BM_START_CUDA_ARRAY_EVENT(caption, descriptor, stream, index, max_elements) \
	}

#define BM_STOP_CUDA_ARRAY_EVENT_LEV(lev, caption, descriptor, stream, index, max_elements)																		\
	if(lev>1){ \
		BM_STOP_CUDA_ARRAY_EVENT(caption, descriptor, stream, index, max_elements) \
	}

#else
#define BM_START_CUDA_ARRAY_EVENT(caption, descriptor, stream, index, max_elements)
#define BM_STOP_CUDA_ARRAY_EVENT(caption, descriptor, stream, index, max_elements)
#define BM_START_CUDA_ARRAY_EVENT_LEV(lev, caption, descriptor, stream, index, max_elements)
#define BM_STOP_CUDA_ARRAY_EVENT_LEV(lev, caption, descriptor, stream, index, max_elements)
#endif

#endif /* BENCHMARK_CUH_ */
