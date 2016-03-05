/* libcdalog.c  ---  cd into Archive, logging facility
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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/timeb.h>
#include <sys/types.h>
#include <execinfo.h>

#include "cdalog.h"

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
