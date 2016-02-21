/* cda_log.h  ---  cd into Archive, logging facility
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

#ifndef CDA_LOG_H_
#define CDA_LOG_H_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/timeb.h>
#include <sys/types.h>

#include <execinfo.h>

#include "cda.h" /* for the color definitions */

struct timeb timeb_s;
char _cda_logbuf[4096] = {0};

void _CDA_BACKTRACE (void);

/* backend function */
void
_CDA_LOG_CORE (char level,
               struct timeb * timebp,
               pid_t pid,
               __typeof__(__FILE__) file,
               __typeof__(__LINE__) line,
		       __typeof__(__FUNCTION__) func,
               char * msgstring) 
{
	ftime (timebp);
	struct tm * ptm = gmtime (&timebp->time);
	fprintf (stderr, (level=='I')?CDA_COLOR_GREEN_B
			:(level=='W')?CDA_COLOR_YELLOW_B
			:(level=='E')?CDA_COLOR_RED_B
			:(level=='D')?CDA_COLOR_CYAN_B
			:CDA_COLOR_RESET);
	fprintf (stderr, "%1c%02d%02d %02d:%02d:%02d.%03d %05d %s:%d] @%s() %s", level,
		   	ptm->tm_mon, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec,
		   	timebp->millitm, pid, file, line, func, msgstring);
	fprintf (stderr, CDA_COLOR_RESET);
	return;
}

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

/* see backtrace(3) */
#define CDA_BT_SIZE 16
void
_CDA_BACKTRACE (void)
{
	int nptrs;
	void * bt_buffer[CDA_BT_SIZE];

	nptrs = backtrace (bt_buffer, CDA_BT_SIZE);
	LOG_INFOF ("backtrace depth %d\n", nptrs);

	backtrace_symbols_fd (bt_buffer, nptrs, STDERR_FILENO);
	return;
}
#undef CDA_BT_SIZE

#undef  CDA_BT_SIZE
#endif /* CDA_LOG_H_ */
