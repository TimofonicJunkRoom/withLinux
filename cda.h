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

#include <archive.h>
#include <archive_entry.h>

#define CDA_VERSION   ("1.6.2 (02 Mar. 2016)")

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
#define CDA_COLOR_BLUE      ((const char *)"\x1B[34m")
#define CDA_COLOR_BLUE_B    ((const char *)"\x1B[34;1m")
#define CDA_COLOR_PURPLE    ((const char *)"\x1B[35m")
#define CDA_COLOR_PURPLE_B  ((const char *)"\x1B[35;1m")
#define CDA_COLOR_CYAN      ((const char *)"\x1B[36m")
#define CDA_COLOR_CYAN_B    ((const char *)"\x1B[36;1m")
#define CDA_COLOR_WHITE     ((const char *)"\x1B[37m")
#define CDA_COLOR_WHILE_B   ((const char *)"\x1B[37;1m")
#define CDA_COLOR_RESET   ((const char *)"\x1B[m")

#endif /* CDA_H_ */
