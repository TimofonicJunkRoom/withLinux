/* C.D.Luminate <cdluminate@163.com> */
/* MIT LICENCE */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <openssl/md5.h>
#include <omp.h>

int
main (int argc, char **argv)
{
	char md_ans[16];
	//bzero (md_ans, 16);
	//MD5 ((const unsigned char *)argv[1], strlen(argv[1]), (unsigned char *)md_ans);
	
	/*char md_ans[16] = {  this is md5 of '0000' 
			   0x4a, 0x7d, 0x1e, 0xd4,
			   0x14, 0x47, 0x4e, 0x40,
			   0x33, 0xac, 0x29, 0xcc,
			   0xb8, 0x65, 0x3d, 0x9b }; */

	int fd = open ("aaaa.md5", O_RDONLY);
	if (fd == -1) {
		perror ("open");
		exit (1);
	}
	read (fd, md_ans, 16);
	
	//char md[16];
	//char c[4];
	//bzero (c, 4);

	char c0 = 0;
	char c1 = 0;
	char c2 = 0;
	char c3 = 0;

	//if (argc != 2)
	//	exit (-1);


	//#pragma omp parallel for num_threads(4) shared(c1,c2,c3) private(x,y,z,w)  
	//#pragma omp parallel for num_threads(4) private(c1,c2,c3)
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
						//printf ("%c%c%c%c           \n", c0, c1, c2, c3);
						write (2, c, 4);
						write (2, "\n", 2);
						exit (0);
					}
				}
			}
		}
	}

	return 0;
}
