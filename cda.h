/* cda.h  ---  cd into Archive
Copyright (C) 2015 Lumin <cdluminate@gmail.com>
License: GPL-3.0+
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 .
 This package is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 GNU General Public License for more details.
 .
 You should have received a copy of the GNU General Public License
 along with this program. If not, see <http://www.gnu.org/licenses/>.
 .
 On Debian systems, the complete text of the GNU General
 Public License version 3 can be found in "/usr/share/common-licenses/GPL-3".
*/

#ifndef CDA_H_
#define CDA_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>

#include <archive.h>
#include <archive_entry.h>

#define CDA_VERSION   ("1.5 (19 Feb. 2016)")

#define CDA_LIST         (0x0001)
#define CDA_EXTRACT      (0x0010)
#define CDA_SHELL        (0x0100)

#define PREFIX        ("/tmp/")
#define TEMPLATE      ("cda.XXXXXX")
#define SHELL         ("/bin/bash")

#define CDA_COLOR_RED       ((const char *)"\x1B[31m")
#define CDA_COLOR_RED_B     ((const char *)"\x1B[31;1m")
#define CDA_COLOR_GREEN     ((const char *)"\x1B[32m")
#define CDA_COLOR_GREEN_B   ((const char *)"\x1B[32;1m")
#define CDA_COLOR_YELLOW    ((const char *)"\x1B[33m")
#define CDA_COLOR_YELLOW_B  ((const char *)"\x1B[33;1m")
#define CDA_COLOR_RESET   ((const char *)"\x1B[m")

/*
 * Wrapper functions 
 */

char *
Getcwd (char * buf, size_t size) {
	char * ret = getcwd (buf, size);
	if (NULL == ret) {
		perror ("getcwd");
		exit (EXIT_FAILURE);
	}
	return ret;
}

int
Stat (char * pathname, struct stat * buf) {
	int ret = stat (pathname, buf);
	if (0 != ret) {
		perror ("stat");
		exit (EXIT_FAILURE);
	}
	return ret;
}

int
Access (char * pathname, int mode) {
	int ret = access (pathname, mode);
	if (0 != ret) {
		perror ("access");
		exit (EXIT_FAILURE);
	}
	return ret;
}

char *
Mkdtemp (char * template) {
	char * ret = mkdtemp (template);
	if (NULL == ret) {
		perror ("mkdtemp");
		exit (EXIT_FAILURE);
	}
	return ret;
}

int
Chdir (const char *path) {
	int ret = chdir (path);
	if (0 != ret) {
		perror ("chdir");
		exit (EXIT_FAILURE);
	}
	return ret;
}

pid_t
Fork (void) {
	pid_t ret = fork ();
	if (-1 == ret) {
		perror ("fork");
		exit (EXIT_FAILURE);
	}
	return ret;
}

pid_t
Waitpid (pid_t pid, int *status, int options) {
	pid_t ret = waitpid (pid, status, options);
	if (-1 == ret) {
		perror ("waitpid");
		exit (EXIT_FAILURE);
	}
	return ret;
}

void *
Malloc (size_t size) {
	void * ret = malloc (size);
	if (NULL == ret) {
		perror ("malloc");
		exit (EXIT_FAILURE);
	}
	return ret;
}

#endif /* CDA_H_ */
