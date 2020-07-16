/*
 * app_log.h
 *
 *  Created on: Aug 14, 2017
 *      Author: stehle
 */

#ifndef APP_LOG_H_
#define APP_LOG_H_



/********************************
 * VERBOSITY LEVELS				*
 ********************************/
// Limits logging to essential messages, such as you would have for a production-like web server
#define APPLOG_VERBOSITY_LEVEL_PRODUCTION 	0

// More verbose logging, such as you would have when running tests
#define APPLOG_VERBOSITY_LEVEL_TEST 		1

// Very verbose logging, such as you would want to help you track down bugs
#define APPLOG_VERBOSITY_LEVEL_DEBUG 		2

// Most verbose logging. Typically only reasonable for debugging individual components to track down bugs.
#define APPLOG_VERBOSITY_LEVEL_DETAILED 	3


/********************************
 * USED VERBOSITY LEVEL			*
 ********************************/
#define APPLOG_VERBOSITY_LEVEL 1

// LOGh
#define APPLOG(level, msg, ...)	\
if(level <= APPLOG_VERBOSITY_LEVEL){ 		\
	printf(msg "\n", ## __VA_ARGS__);\
}


#define APPLOG_PRODUCTION(msg, ...)	APPLOG(APPLOG_VERBOSITY_LEVEL_PRODUCTION, msg, ##__VA_ARGS__);
#define APPLOG_TEST(msg, ...)		APPLOG(APPLOG_VERBOSITY_LEVEL_TEST, msg, ##__VA_ARGS__);
#define APPLOG_DEBUG(msg, ...)		APPLOG(APPLOG_VERBOSITY_LEVEL_DEBUG, msg, ##__VA_ARGS__);
#define APPLOG_DETAILED(msg, ...)	APPLOG(APPLOG_VERBOSITY_LEVEL_DETAILED, msg, ##__VA_ARGS__);

/********************************
 * PRINTING BINARY REPRESENTATION
 ********************************/
#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c %c%c%c%c%c%c%c%c %c%c%c%c%c%c%c%c %c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  ((byte) & 0x80 ? '1' : '0'), \
  ((byte) & 0x40 ? '1' : '0'), \
  ((byte) & 0x20 ? '1' : '0'), \
  ((byte) & 0x10 ? '1' : '0'), \
  ((byte) & 0x08 ? '1' : '0'), \
  ((byte) & 0x04 ? '1' : '0'), \
  ((byte) & 0x02 ? '1' : '0'), \
  ((byte) & 0x01 ? '1' : '0')


#define UINT_TO_BINARY(uint)  \
	BYTE_TO_BINARY((uint>>24)&0xFF),\
	BYTE_TO_BINARY((uint>>16)&0xFF), \
	BYTE_TO_BINARY((uint>>8)&0xFF), \
	BYTE_TO_BINARY(uint&0xFF)

#endif /* APP_LOG_H_ */
