/* a8freq :: a8lu.c
 *      l(ower)u(pper)
 * 	convert alphabets between upper and lower case.
 *
 * Author : C.D.Luminate 
 * 	< cdluminate AT gmail DOT com >
 *	started at 02 / 06 /2014
 *
 * https://github.com/CDLuminate/a8freq
 */

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

void Usage (char *prog_name)
{
	fprintf (stderr, "\
Usage : %s [-r] [FILE]\n\
Convert alphabets between upper and lower case\n\
If FILE not specified, stdin will be used.\n\
       -r    reverse converting, upper to lower.\n",
       		 prog_name);
}


int
main (int argc, char *argv[])
{
	FILE *in_file = stdin;
	FILE *out_file = stdout;

	/* defalut is lower -> upper, revflag is reverse_flag */
	int opt, revflag;
	register int buf = 0;
	revflag = 0;

	while ((opt = getopt (argc, argv, "r")) != -1) {
		switch (opt) {
			case 'r':
				revflag = 1;
				break;
			default:
				Usage (argv[0]);
				exit (EXIT_FAILURE);
		}
	}

	/* tell if a FILE is specified */
	if (optind < argc) {
		if ((in_file = fopen (argv[optind], "r")) == NULL) {
			perror ("fopen");
			exit (EXIT_FAILURE);
		}
	}

	while ( (buf = fgetc (in_file)) != EOF && !feof(in_file)) {
		/* change case according to revflag */
		switch (buf) {
			case 'a' ... 'z':
				if (!revflag) buf -= 32;
				break;
			case 'A' ... 'Z':
				if (revflag) buf += 32;
				break;
			default:
				break;
		}
		
		fputc (buf, out_file);
	}

	fclose (in_file);
	fclose (out_file);
	return 0;
}
