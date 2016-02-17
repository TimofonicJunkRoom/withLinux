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

#include "cda.h" /* for the color definitions */

struct timeb timeb_s;
char _cda_logbuf[4096] = {0};

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
			:CDA_COLOR_RESET);
	fprintf (stderr, "%1c%02d%02d %02d:%02d:%02d.%03d %05d %s:%d @%s()] %s", level,
		   	ptm->tm_mon, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec,
		   	timebp->millitm, pid, file, line, func, msgstring);
	fprintf (stderr, CDA_COLOR_RESET);
	return;
}

#define LOG_INFO(_cda_msg) do { \
	_CDA_LOG_CORE ('I', &timeb_s, getpid(), __FILE__, __LINE__, __FUNCTION__, ((_cda_msg))); \
} while (0)

#define LOG_WARN(_cda_msg) do { \
	_CDA_LOG_CORE ('W', &timeb_s, getpid(), __FILE__, __LINE__, __FUNCTION__, ((_cda_msg))); \
} while (0)

#define LOG_ERROR(_cda_msg) do { \
	_CDA_LOG_CORE ('E', &timeb_s, getpid(), __FILE__, __LINE__, __FUNCTION__, ((_cda_msg))); \
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

#endif /* CDA_LOG_H_ */
