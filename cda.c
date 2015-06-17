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

#define PREFIX "/tmp/"
#define TEMPLATE "./cda.XXXXXX"
#define SHELL "bash"

int debug = 1;
int force = 0;

char * prefix = PREFIX;
char * archive;

int           status;
pid_t         pid;
struct stat * stat_buf;
char        * path_buf;
char        * cmd_buf;
char          template[] = TEMPLATE;
char        * temp_dir;
char        * newargv[] = { NULL, NULL, NULL, NULL,
	                        NULL, NULL, NULL, NULL }; /* should be enough? */
char        * newenv[] = { NULL };
char        * buffer; /* general purpose */

void
Usage (char *myname)
{
	printf (""
"          cda - chdir into Archive\n"
"Usage:\n"
"    %s <ARCHIVE> [-f]\n"
"Option:\n"
"    -f        force remove tmpdir, instead of interactive rm.\n"
"    -d <TEMP> Specify the temp directory to use.\n"
"              (would override the CDA env).\n"
"Environment:\n"
"    CDA   specify the temp directory to use.\n"
"          (default: /tmp)\n"
"Formats:\n"
"    tar.gz | tgz, tar.xz | txz, \n"
"    tar.bz2 | tbz | tbz2, tar, zip | jar, 7z\n"
"Version:\n"
"    %s\n"
"", myname, myversion);
	return;
}

int
flush_newargv (char ** _newargv)
{
	int i;
	for (i = 0; i < 8; i++)
		_newargv[i] = NULL;
	return 0;
}

char **
setargvl8 (char **_newargv,
		   char *v0, char *v1, char *v2, char *v3,
		   char *v4, char *v5, char *v6, char *v7)
{
	/* set newargv length 8 */
	flush_newargv (_newargv);
	newargv[0] = v0;
	newargv[1] = v1;
	newargv[2] = v2;
	newargv[3] = v3;
	newargv[4] = v4;
	newargv[5] = v5;
	newargv[6] = v6;
	newargv[7] = NULL; /* last one must be NULL, char *v7 overriden */	
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
		setargvl8 (newargv, "rm", "-rf", _tmpdir,
				   (1 == _force)?(NULL):("-i"), NULL, NULL, NULL, NULL);
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
	/* for getopt() */
	int opt;
	/* check argc */
	if (2 > argc) {
		printf ("! Please Spefify Archive.\n");
		Usage (argv[0]);
		exit (EXIT_FAILURE);
	}
	/* check env and apply env CDA */
	if (NULL == getenv("CDA")) {
		if (1<debug) perror ("getenv");
	} else {
		prefix = getenv("CDA");
		if (debug) printf ("* CDA = \"%s\"\n", prefix);
	}
	/* parse argument */
	while ((opt = getopt(argc, argv, "fd:")) != -1) {
		switch (opt) {
		case 'f': /* force */
			force = 1;
			break;
		case 'd': /* destination */
			prefix = optarg;
			break;
		default:
			fprintf (stderr, "! Error: option not defined.\n");
			exit (EXIT_FAILURE);
		}
	}
	if (optind >= argc) {
		fprintf (stderr, "! Error: No Archive.\n");
		exit (EXIT_FAILURE);
	}
	archive = argv[optind];
	if (debug) fprintf (stderr,
			"* Force = %d, Archive = %s\n",
			force, archive);
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
	if (stat (archive, stat_buf) == -1) {
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
		printf ("* Extract Archive \"%s\"\n", archive);
	} else {
		printf ("* No, I don't manipulate directory or something else.\n");
		exit (EXIT_SUCCESS);
	}
	/* check mode of target */
	if (access (archive, R_OK) == -1) {
		perror ("access");
		exit (EXIT_FAILURE);
	}
	if (1<debug) perror ("access");
	/* chdir to prefix */
	if (-1 == chdir (prefix)) {
		perror ("chdir");
		exit (EXIT_FAILURE);
	}
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
		setargvl8 (newargv, TAR, NULL, archive, "-C", temp_dir, NULL, NULL, NULL);
		if (strstr(archive, ".tar.gz") != NULL ||
		    strstr(archive, ".tgz")    != NULL) {
			if (debug) printf ("* detected [ .tar.gz | .tgz ]\n");
			newargv[1] = "zxf";
		} else if (strstr(archive, ".tar.bz2") != NULL ||
				   strstr(archive, ".tbz2")    != NULL ||
				   strstr(archive, ".tbz")     != NULL) {
			if (debug) printf ("* detedted [ .tar.bz2 | .tbz | .tbz2 ]\n");
			newargv[1] = "jxf";
		} else if (strstr(archive, ".tar.xz") != NULL ||
				   strstr(archive, ".txz")    != NULL) {
			if (debug) printf ("* detedted [ .tar.xz | .txz ]\n");
			newargv[1] = "Jxf";
		} else if (strstr(archive, ".tar") != NULL) {
			if (debug) printf ("* detedted [ .tar ]\n");
			newargv[1] = "xf";
		} else if (strstr(archive, ".zip") != NULL ||
				   strstr(archive, ".jar") != NULL) {
			if (debug) printf ("* detedted [ .zip | .jar ]\n");
			flush_newargv (newargv);
			decompress = UNZIP;
			decompress_fname = UNZIP_fname;
			setargvl8 (newargv, UNZIP, "-q", archive, "-d", temp_dir, NULL, NULL, NULL);
		} else if(strstr(archive, ".7z") != NULL) {
			if (debug) printf ("* detedted [ .7z ]\n");
			bzero (buffer, 4096);
			flush_newargv (newargv);
			decompress = _7Z;
			decompress_fname = _7Z_fname;
			setargvl8 (newargv, _7Z, "x", archive,
					   strncat(strncat(buffer, "-o", 4095), temp_dir, 4095),
					   NULL, NULL, NULL, NULL);
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
	system (SHELL);
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
