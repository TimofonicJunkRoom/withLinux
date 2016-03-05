/* cdalog.h  ---  cd into Archive, logging facility header
 * Copyright (C) 2015 Lumin <cdluminate@gmail.com>
 * License: GPL-3.0+
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * .
 * This package is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details.
 * .
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 * .
 * On Debian systems, the complete text of the GNU General
 * Public License version 3 can be found in "/usr/share/common-licenses/GPL-3".
 */

#ifndef LIBCDALOG_H_
#define LIBCDALOG_H_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/timeb.h>
#include <sys/types.h>
#include <execinfo.h>

struct timeb timeb_s;
char _cda_logbuf[4096];

#define CDA_COLOR_RED       ((const char *)"\x1B[31m")
#define CDA_COLOR_RED_B     ((const char *)"\x1B[31;1m")
#define CDA_COLOR_GREEN     ((const char *)"\x1B[32m")
#define CDA_COLOR_GREEN_B   ((const char *)"\x1B[32;1m")
#define CDA_COLOR_YELLOW    ((const char *)"\x1B[33m")
#define CDA_COLOR_YELLOW_B  ((const char *)"\x1B[33;1m")
#define CDA_COLOR_BLUE      ((const char *)"\x1B[34m")
#define CDA_COLOR_BLUE_B    ((const char *)"\x1B[34;1m")
#define CDA_COLOR_PURPLE    ((const char *)"\x1B[35m")
#define CDA_COLOR_PURPLE_B  ((const char *)"\x1B[35;1m")
#define CDA_COLOR_CYAN      ((const char *)"\x1B[36m")
#define CDA_COLOR_CYAN_B    ((const char *)"\x1B[36;1m")
#define CDA_COLOR_WHITE     ((const char *)"\x1B[37m")
#define CDA_COLOR_WHILE_B   ((const char *)"\x1B[37;1m")
#define CDA_COLOR_RESET     ((const char *)"\x1B[m")

void _CDA_BACKTRACE (void);
void _CDA_LOG_CORE (char level, struct timeb * timebp, pid_t pid,
               __typeof__(__FILE__) file,
               __typeof__(__LINE__) line,
		       __typeof__(__FUNCTION__) func,
               char * msgstring);

#define LOG_DEBUG(_cda_msg) do { \
	_CDA_LOG_CORE ('D', &timeb_s, getpid(), __FILE__, __LINE__, __FUNCTION__, ((_cda_msg))); \
} while (0)

#define LOG_INFO(_cda_msg) do { \
	_CDA_LOG_CORE ('I', &timeb_s, getpid(), __FILE__, __LINE__, __FUNCTION__, ((_cda_msg))); \
} while (0)

#define LOG_WARN(_cda_msg) do { \
	_CDA_LOG_CORE ('W', &timeb_s, getpid(), __FILE__, __LINE__, __FUNCTION__, ((_cda_msg))); \
} while (0)

#define LOG_ERROR(_cda_msg) do { \
	_CDA_LOG_CORE ('E', &timeb_s, getpid(), __FILE__, __LINE__, __FUNCTION__, ((_cda_msg))); \
	_CDA_BACKTRACE (); \
} while (0)

#define LOG_DEBUGF(...) do { \
	snprintf (_cda_logbuf, 4095, ##__VA_ARGS__); \
	LOG_DEBUG (_cda_logbuf); \
} while (0)

#define LOG_INFOF(...) do { \
	snprintf (_cda_logbuf, 4095, ##__VA_ARGS__); \
	LOG_INFO (_cda_logbuf); \
} while (0)

#define LOG_WARNF(...) do { \
	snprintf (_cda_logbuf, 4095, ##__VA_ARGS__); \
	LOG_WARN (_cda_logbuf); \
} while (0)

#define LOG_ERRORF(...) do { \
	snprintf (_cda_logbuf, 4095, ##__VA_ARGS__); \
	LOG_ERROR (_cda_logbuf); \
} while (0)

#endif /* LIBCDALOG_H_ */
