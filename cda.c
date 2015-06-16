/* cda.c  ---  cd into Archive
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

#include "cda.h"
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <string.h>

char * TAR = "tar";
char * RM = "rm";

#define PREFIX ""
#define TEMPLATE "/tmp/cda.XXXXXX"

int debug = 1;
int force = 0;

int           status;
pid_t         pid;
struct stat * stat_buf;
char        * path_buf;
char        * cmd_buf;
char          template[] = TEMPLATE;
char        * temp_dir;
char        * newargv[] = { NULL, NULL, NULL, NULL, NULL, NULL };
char        * newenv[] = { NULL };

void
Usage (char *myname)
{
	printf (""
"Usage:\n"
"    %s  <ARCHIVE> [-f]\n"
"Option:\n"
"    -f force remove tmpdir, instead of interactive rm.\n"
"", myname);
	return;
}

int
remove_tmpdir (char * _tmpdir, int _force, int _verbose)
{   
	int _tmp = 0x00;
	/* remove temp_dir */
	if ((pid = fork()) == -1) {
		perror ("fork");
		exit (EXIT_FAILURE);
	}
	if (pid == 0) {  /* fork : child */
		/* construct newargv for rm */
		newargv[0] = "rm"; 
		newargv[1] = "-rf";
		newargv[2] = _tmpdir; 
		newargv[3] = (1 == _force) ? (NULL) : ("-i");
		newargv[4] = NULL; 
		newargv[5] = NULL; /* this is the last one ! */
		/* end constructing newargv */
		execve ("/bin/rm", newargv, newenv);
		perror ("execve"); /* execve only returns on error */
		exit (EXIT_FAILURE);
	} else {  /* fork : parent */
		if (debug) printf ("* fork() [%d]\n", pid);
		waitpid (-1, &_tmp, 0);
		if (debug) printf ("* child terminated (%d).\n", _tmp);
		if (0 != _tmp) {
			printf ("* cda: rm failed. (%d)\n", _tmp);
			exit (EXIT_FAILURE);
		}
	}
	return _tmp;
}

int
main (int argc, char **argv, char **env)
{
	/* check argv[1] */
	if (argv[1] == NULL) {
		Usage (argv[0]);
		exit (EXIT_FAILURE);
	}
	/* parse argv[2] */
	if (NULL != argv[2])
		if (NULL != strstr(argv[2], "-f"))
			force = 1;
	/* malloc buffers, check NULL  */
	stat_buf = malloc (sizeof(struct stat));
	path_buf = malloc (4096);
	cmd_buf  = malloc (4096);
	if (stat_buf == NULL || path_buf == NULL || cmd_buf == NULL) {
		printf ("! cda: malloc failed\n");
		exit (EXIT_FAILURE);
	}

	/* stat the target (archive)file/dir */
	if (stat (argv[1], stat_buf) == -1) {
		perror ("stat");
		exit (EXIT_FAILURE);
	}
	if (debug) {
		perror ("stat");
		printf ("* stat_buf: uid= %d; gid= %d; mode= %o;\n",
				stat_buf -> st_uid,
				stat_buf -> st_gid,
				stat_buf -> st_mode);
	}
	/* check target archive */
	if ( stat_buf -> st_mode & S_IFREG ) {
		printf ("* Extract Archive \"%s\"\n", argv[1]);
	} else {
		printf ("* No, I don't manipulate directory or something else.\n");
		exit (EXIT_SUCCESS);
	}
	/* check mode of target */
	if (access (argv[1], R_OK) == -1) {
		perror ("access");
		exit (EXIT_FAILURE);
	}
	if (debug) perror ("access");
	/* mkdtemp for extracting files */
	if (debug) printf ("* using template \"%s\"\n", template);
	if ((temp_dir = mkdtemp (template)) == NULL) {
		printf ("! mkdtemp() failed.\n");
		exit (EXIT_FAILURE);
	}
	if (debug) printf ("* created temp dir \"%s\"\n", temp_dir);
	/* extract archive into temp_dir */
	if ((pid = fork()) == -1) {
		perror ("fork");
		exit (EXIT_FAILURE);
	}
	if (pid == 0) {
		/* fork : child */
		if ((strstr(argv[1], ".tar.gz") != NULL)||(strstr(argv[1], ".tgz") != NULL)) {
			if (debug) printf ("* detected [ .tar.gz | .tgz ]\n");
			//newargv = { "tar", "zxvf", argv[1], "-C", temp_dir, NULL };
			newargv[1] = "zxf";
		} else if (strstr(argv[1], ".tar.bz2") != NULL) {
			if (debug) printf ("* detedted [ .tar.bz2 ]\n");
			//newargv = { "tar", "jxvf", argv[1], "-C", temp_dir, NULL };
			newargv[1] = "jxf";
		} else if (strstr(argv[1], ".tar.xz") != NULL) {
			//newargv = { "tar", "Jxvf", argv[1], "-C", temp_dir, NULL };
			newargv[1] = "Jxf";
		} else if (strstr(argv[1], ".tar") != NULL) {
			//newargv = { "tar", "xvf", argv[1], "-C", temp_dir, NULL };
			newargv[1] = "xf";
		} else {
			printf ("* I finally realized that, you are not feeding me an Archive !\n");
			exit (EXIT_SUCCESS);
		}
		/* FYI: tar zxvf x.tar.gz -C /tmp NULL */
		newargv[0] = TAR;
		newargv[2] = argv[1];
		newargv[3] = "-C";
		newargv[4] = temp_dir;
		newargv[5] = NULL; /* this is the last one ! */
		/* end constructing newargv */
		execve ("/bin/tar", newargv, newenv);
		perror ("execve");
		exit (EXIT_FAILURE);
	} else {
		/* fork : parent */
		if (debug) printf ("* p: fork() [%d]\n", pid);
		waitpid (-1, &status, 0);
		if (debug) printf ("* child terminated (%d).\n", status);
		if (0 != status) {
			printf ("* child exited with error (%d).\n", status);
			exit (EXIT_FAILURE);
		}
	}
	/* step into temp and popup a shell */
	if (debug) printf ("* step into tempdir %s\n", temp_dir);
	if (chdir(temp_dir) == -1) {
		perror ("chdir");
		exit (EXIT_FAILURE);
	}
	if (NULL == getcwd (path_buf, 4095)) {
		printf ("* getcwd failed\n");
		exit (EXIT_FAILURE);
	}
	if (debug) printf ("* now pwd = %s\n", path_buf);
	system ("bash");	

	/* remove the temp dir */
	printf ("* cda: OK, removing temp directory \"%s\"...\n", path_buf);
	/* traditional delete
    snprintf (cmd_buf, 4095, "cd /; %s -i -rf %s", RM, path_buf);
	if (debug) printf ("* run \"%s\"\n", cmd_buf);
	system (cmd_buf); */
	remove_tmpdir (path_buf, force, 0);
	/* XXX: don't forget to free() ! */
	free (cmd_buf);
	free (path_buf);
	free (stat_buf);
	//close (fd);
	return 0;
}

