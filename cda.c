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

extern char * myversion;

char * decompress;
char * decompress_fname;

char * UNZIP = "unzip";
char * UNZIP_fname = "/usr/bin/unzip";
char * TAR = "tar";
char * TAR_fname = "/bin/tar";
char * _7Z = "7z";
char * _7Z_fname = "/usr/bin/7z";

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
char        * buffer; /* general purpose */

void
Usage (char *myname)
{
	printf (""
"          cda - chdir into Archive\n"
"Usage:\n"
"    %s  <ARCHIVE> [-f]\n"
"Option:\n"
"    -f force remove tmpdir, instead of interactive rm.\n"
"\n"
"  supports: tar.gz|tgz, tar.xz|txz, tar.bz2|tbz|tbz2, tar, zip, 7z"
"  version: %s\n"
"", myname, myversion);
	return;
}

int
flush_newargv (char ** _newargv)
{
	int i;
	for (i = 0; i < 6; i++)
		_newargv[i] = NULL;
	return 0;
}

char **
setargvl6 (char **_newargv, char *v0, char *v1, char *v2, char *v3, char *v4, char *v5)
{
	/* set newargv length 6 */
	flush_newargv (_newargv);
	newargv[0] = v0;
	newargv[1] = v1;
	newargv[2] = v2;
	newargv[3] = v3;
	newargv[4] = v4;
	newargv[5] = NULL; /* last one must be NULL, char *v5 overriden */	
	return _newargv;
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
		setargvl6 (newargv, "rm", "-rf", _tmpdir,
				   (1 == _force)?(NULL):("-i"), NULL, NULL);
		/* end constructing newargv */
		execve ("/bin/rm", newargv, newenv);
		perror ("execve"); /* execve only returns on error */
		exit (EXIT_FAILURE);
	} else {  /* fork : parent */
		if (1<debug) printf ("* fork() [%d] to execve rm\n", pid);
		waitpid (-1, &_tmp, 0);
		if (debug) printf ("* Child RM terminated with (%d) - %s.\n", _tmp,
				           (0==_tmp)?"Success":"Failure");
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
	/* TODO: use getopt() ? */
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
	buffer   = malloc (4096);
	if (stat_buf == NULL || path_buf == NULL || cmd_buf == NULL || buffer == NULL) {
		printf ("! cda: malloc failed\n");
		exit (EXIT_FAILURE);
	}

	/* stat the target (archive)file/dir */
	if (stat (argv[1], stat_buf) == -1) {
		perror ("stat");
		exit (EXIT_FAILURE);
	}
	if (1<debug) {
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
	if (1<debug) perror ("access");
	/* mkdtemp for extracting files */
	if (1<debug) printf ("* using template \"%s\"\n", template);
	if ((temp_dir = mkdtemp (template)) == NULL) {
		printf ("! mkdtemp() failed.\n");
		exit (EXIT_FAILURE);
	}
	if (debug) printf ("* Created temp dir \"%s\"\n", temp_dir);
	/* extract archive into temp_dir */
	decompress = TAR; /* use tar by default */
	decompress_fname = TAR_fname;
	if ((pid = fork()) == -1) {
		perror ("fork");
		exit (EXIT_FAILURE);
	}
	if (pid == 0) { /* fork : child */
		/* FYI: tar zxvf x.tar.gz -C /tmp NULL */
		setargvl6 (newargv, TAR, NULL, argv[1], "-C", temp_dir, NULL);
		if (strstr(argv[1], ".tar.gz") != NULL ||
		    strstr(argv[1], ".tgz")    != NULL) {
			if (debug) printf ("* detected [ .tar.gz | .tgz ]\n");
			newargv[1] = "zxf";
		} else if (strstr(argv[1], ".tar.bz2") != NULL ||
				   strstr(argv[1], ".tbz2")    != NULL ||
				   strstr(argv[1], ".tbz")     != NULL) {
			if (debug) printf ("* detedted [ .tar.bz2 | .tbz | .tbz2 ]\n");
			newargv[1] = "jxf";
		} else if (strstr(argv[1], ".tar.xz") != NULL ||
				   strstr(argv[1], ".txz")    != NULL) {
			if (debug) printf ("* detedted [ .tar.xz | .txz ]\n");
			newargv[1] = "Jxf";
		} else if (strstr(argv[1], ".tar") != NULL) {
			if (debug) printf ("* detedted [ .tar ]\n");
			newargv[1] = "xf";
		} else if (strstr(argv[1], ".zip") != NULL) {
			if (debug) printf ("* detedted [ .zip ]\n");
			flush_newargv (newargv);
			decompress = UNZIP;
			decompress_fname = UNZIP_fname;
			setargvl6 (newargv, UNZIP, "-q", argv[1], "-d", temp_dir, NULL);
		} else if(strstr(argv[1], ".7z") != NULL) {
			if (debug) printf ("* detedted [ .7z ]\n");
			bzero (buffer, 4096);
			flush_newargv (newargv);
			decompress = _7Z;
			decompress_fname = _7Z_fname;
			setargvl6 (newargv, _7Z, "x", argv[1],
					   strncat(strncat(buffer, "-o", 4095), temp_dir, 4095),
					   NULL, NULL);
		} else {
			/* TODO: more formats ? */
			printf ("* I don't recogonize this kind of \"Archive\" !\n");
			exit (EXIT_FAILURE);
		}
		/* end constructing newargv */
		execve (decompress_fname, newargv, newenv);
		perror ("execve");
		exit (EXIT_FAILURE);
	} else { /* fork : parent */
		if (1<debug) printf ("* fork() [%d] to execve tar\n", pid);
		waitpid (-1, &status, 0);
		if (debug) printf ("* Child Decompressor terminated (%d).\n",
				           status);
		if (0 != status) {
			printf ("* child tar exited with error (%d).\n", status);
			exit (EXIT_FAILURE);
		}
	}
	/* step into temp and popup a shell */
	if (debug) printf ("* Stepping into Archive (tempdir): %s\n", temp_dir);
	if (chdir(temp_dir) == -1) {
		perror ("chdir");
		exit (EXIT_FAILURE);
	}
	if (NULL == getcwd (path_buf, 4095)) {
		printf ("* getcwd failed\n");
		exit (EXIT_FAILURE);
	}
	if (debug) printf ("* cda: PWD = %s\n*      fork and execve bash ...\n", path_buf);
	/* TODO: fork a new one to execve bash ? */
	system ("bash");
	/* when user exited bash above, this program continues from here */

	/* remove the temp dir */
	printf ("* cda: OK, Removing temp directory \"%s\"...\n", path_buf);
	/* traditional delete
    snprintf (cmd_buf, 4095, "cd /; %s -i -rf %s", RM, path_buf);
	if (debug) printf ("* run \"%s\"\n", cmd_buf);
	system (cmd_buf); */
	remove_tmpdir (path_buf, force, 0);
	/* XXX: don't forget to free() ! */
	free (buffer);
	free (cmd_buf);
	free (path_buf);
	free (stat_buf);
	return 0;
}
