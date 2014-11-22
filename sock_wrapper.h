/* sock_wrapper.h
 * data cruncher for Bytefreq, this is a part of bytefreq

   Count Byte/Char freqency, using Serial/Parallel Approaches.

   C.D.Luminate <cdluminate AT 163 DOT com> 
   MIT Licence, 2014
*/

#include <stdio.h>
#include <stdlib.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

/* NOTE : all functions in this file is wrapper function */

//        int socket(int domain, int type, int protocol);
int
Socket (int domain, int type, int protocol)
{
	int _ret;
	if ((_ret = socket(domain, type, protocol)) == -1){
		perror ("socket");
		exit (EXIT_FAILURE);
	}
	return _ret;
}

int 
Socketpair (int domain, int type, int protocol, int sv[2])
{
	int _ret;
	if ((_ret = socketpair (domain, type, protocol, sv)) == -1) {
		perror ("socketpair");
		exit (EXIT_FAILURE);
	}
	return _ret;
}

pid_t
Fork (void)
{
	pid_t _ret;
	if ((_ret = fork()) == -1) {
		perror ("fork");
		exit (EXIT_FAILURE);
	}
	return _ret;
}
