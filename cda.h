/* cda.h */

char * myversion = "0.2";

int
remove_tmpdir (char * tmpdir, int interactive, int verbose);

void
Usage (char *myname);

int
flush_newargv (char ** _newargv);

char **
setargvl8 (char **_newargv,
		   char *v0, char *v1, char *v2, char *v3,
		   char *v4, char *v5, char *v6, char *v7);

int
remove_tmpdir (char * _tmpdir, int _force, int _verbose);
