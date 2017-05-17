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
#include <errno.h>
#include <signal.h>

#include <archive.h>
#include <archive_entry.h>

#define CDA_VERSION   ("1.7 (17 May. 2017)")

#define CDA_LIST         (0x0001)
#define CDA_EXTRACT      (0x0010)
#define CDA_SHELL        (0x0100)

#define PREFIX        ("/tmp/")
#define TEMPLATE      ("cda.XXXXXX")

char *
cda_getshell(void)
{
	char * shell = getenv("SHELL");
	if (NULL == shell) {
		shell = "/bin/bash";
	}
	return shell;
}

#endif /* CDA_H_ */
