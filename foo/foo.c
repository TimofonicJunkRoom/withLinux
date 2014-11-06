/* C.D.Luminate <cdluminate@163.com> */
/* MIT LICENCE */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#include <openssl/md5.h>
#include <omp.h>

int
main (int argc, char **argv)
{
	char md_ans[16];
	bzero (md_ans, 16);

	//char md[16];

	//char c[4];
	//bzero (c, 4);

	char c0 = 0;
	char c1 = 0;
	char c2 = 0;
	char c3 = 0;

	if (argc != 2)
		exit (-1);

	MD5 ((const unsigned char *)argv[1], strlen(argv[1]), (unsigned char *)md_ans);

	//#pragma omp parallel for num_threads(4) shared(c1,c2,c3) private(x,y,z,w)  
	#pragma omp parallel for num_threads(4) private(c1,c2,c3)
	for (c0 = 32; c0 < 127; c0++) {
		char c[5];
		bzero (c, 5);
		char md[16];

		for (c1 = 32; c1 < 127; c1++) {
			for (c2 = 32; c2 < 127; c2++) {
				for (c3 = 32; c3 < 127; c3++) {

					c[0] = c0;
					c[1] = c1;
					c[2] = c2;
					c[3] = c3;

					//printf ("%c%c%c%c           \n", c0, c1, c2, c3);
					//write (1, md, 16);

					MD5 ((const unsigned char *)c, 4, (unsigned char *)md);
					if (memcmp (md, md_ans, 16) == 0) {
						printf ("%c%c%c%c           \n", c0, c1, c2, c3);
						write (2, c, 4);
						exit (0);
					}
				}
			}
		}
	}
	

	return 0;
}
