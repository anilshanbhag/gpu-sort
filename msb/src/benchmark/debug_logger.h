#ifndef DEBUG_LOGGER_H_
#define DEBUG_LOGGER_H_

#include "benchmark/get_real_time.h"

#define DEBUG_LEVEL 1
// LOGh
#define LOG(level, msg, ...)	\
if(level <= DEBUG_LEVEL){ 		\
	printf(msg"\n", ## __VA_ARGS__);\
}

// CPU TIME (START)
#define START_CPU_TIMER(level, identifier) 	\
double start_ ## identifier;		\
char loglev_ ## identifier = level;	\
if(level <= DEBUG_LEVEL){			\
	start_ ## identifier = get_real_time(); \
}
// CPU TIME (GET)
#define GET_CPU_TIMER(identifier, ...) 	\
		(loglev_ ## identifier <= DEBUG_LEVEL ? get_real_time() - start_ ## identifier : 0.0)
// CPU TIME (STOP)
#define STOP_CPU_TIMER(identifier, out, ...) 	\
if(loglev_ ## identifier <= DEBUG_LEVEL){					\
	double after_ ## identifier; 				\
	after_ ## identifier = get_real_time() ;	\
	printf( out" (%9.6f ms) \n", ##__VA_ARGS__, (after_ ## identifier - start_ ## identifier) * 1000.0); \
}

#define RECORD_CUDA_EVENT(event_var, stream) \
		cudaEventCreate(&event_var);\
		cudaEventRecord(event_var, stream);
/*
#define ENQUEUE_CUDA_EVENT(identifier, stream) \
		cudaEvent_t identifier ## _start;\
		cudaEventCreate(&identifier ## _start);\
		cudaEventRecord(identifier ## _start, stream);
#define MEASURE_CUDA_TIME(stream, start_evnt, float_ptr) 		\
		cudaEvent_t start_evnt ## _stop;					\
		cudaEventCreate(&start_evnt ## _stop);				\
		cudaEventRecord( start_evnt ## _stop, stream); 		\
		cudaEventElapsedTime( float_ptr, start_evnt, start_evnt ## _stop ); \

		*/

// CUDA_START/STOP TIMER MACRO
#define START_CUDA_TIMER(debugLevel, identifier, stream) 		\
	unsigned char identifier ##_dbgLevel = debugLevel; 		\
	cudaEvent_t identifier ## _start, identifier ## _stop;	\
	cudaStream_t identifier ## _strm = stream;						\
	float elapsedTime ## identifier;						\
	if(identifier ##_dbgLevel <= DEBUG_LEVEL) {			\
		cudaEventCreate(&identifier ## _start);				\
		cudaEventCreate(&identifier ## _stop);				\
		cudaEventRecord(identifier ## _start, stream); 			\
	}

#define STOP_CUDA_TIMER(identifier, out, ...) \
	if(identifier ##_dbgLevel <= DEBUG_LEVEL) {	\
		cudaEventRecord( identifier ## _stop, identifier ## _strm); \
		cudaEventSynchronize( identifier ## _stop ); \
		cudaEventElapsedTime( &elapsedTime ## identifier, identifier ## _start, identifier ## _stop ); \
		printf( out" (%6.3f ms) \n", ##__VA_ARGS__, elapsedTime ## identifier); \
	}

#endif /* DEBUG_LOGGER_H_ */
