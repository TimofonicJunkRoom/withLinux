/* crunch.c
 * data cruncher for Bytefreq, this is a part of bytefreq

   Count Byte/Char freqency, using Serial/Parallel Approaches.
 
   C.D.Luminate <cdluminate AT 163 DOT com> 
   MIT Licence, 2014
 */


 /* TODO : find a proper buffer size
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>

#include <omp.h>

#include <sys/socket.h>
#include <sys/un.h>
#include "sock_wrapper.h"

/* 131072 Bytes, 128KB buffer */
#define BF_BFSZ_SERI (131072*sizeof(char))
#define BF_BFSZ_PARA (524288*sizeof(char))
#define BF_BFSZ_UNIX (262144*sizeof(char))

long crunch_serial (int _fd, long _counter[256], int _verbose);
long crunch_parallel (int _fd, long _counter[256], int _verbose);
long crunch_unixsock (int _fd, long _counter[256], int _verbose);
void *Malloc (size_t size);

long crunch_serial (int _fd, long _counter[256], int _verbose)
{
	/* the value to return */
	long _total_read = 0;

	/* flush counter */
	bzero (_counter, 256*sizeof(long));

	/* allocate buffer and flush it */
	char *_buf;
	_buf = (char *)Malloc (BF_BFSZ_SERI);
	if (_buf == NULL) exit (1);
	bzero (_buf, BF_BFSZ_SERI);

	/* start crunching */
	int _loop;
	long _readn;
	if (_verbose) write (2, "!", 1);
	while ((_readn = read(_fd, _buf, BF_BFSZ_SERI)) > 0) {
		if (_verbose) write (2, ".", 1);
		_total_read += _readn;
		/* #pragma omp parallel for */
		for (_loop = 0; _loop < _readn; _loop++) {
			_counter[(unsigned char)*(_buf+_loop)]++;
		}
	}
	/* free buffer and return */
	free (_buf);
	if (_verbose) write (2, "!\n", 2);
	return _total_read;
}

long crunch_parallel (int _fd, long _counter[256], int _verbose)
{
	/* the value to return */
	long _total_read = 0;

	/* flush counter */
	bzero (_counter, 256*sizeof(long));

	/* allocate buffer and flush it */
	char *_buf;
	_buf = (char *)Malloc (BF_BFSZ_PARA);
	if (_buf == NULL) exit (1);
	bzero (_buf, BF_BFSZ_PARA);

	/* start crunching */
	int _loop;
	long _readn;
	while ((_readn = read(_fd, _buf, BF_BFSZ_PARA)) > 0) {
		_total_read += _readn;
		#pragma omp parallel for
		for (_loop = 0; _loop < _readn; _loop++) {
			_counter[(unsigned char)*(_buf+_loop)]++;
		}
	}
	/* free buffer and return */
	free (_buf);
	return _total_read;
}

/* ============================================================================ */
long
crunch_unixsock (int _fd, long _counter[256], int _verbose)
{
#define UNIXPATH "/tmp/bytefreq_socket_unix"
	pid_t pid;

	/* prepare variables */
	long _ret_tot = 0;

	int unixfd[2];
	bzero (unixfd, sizeof(unixfd));

	struct sockaddr_un unixsock;
	bzero (&unixsock, sizeof(unixsock));

	/* launch socket */
	Socketpair (AF_UNIX, SOCK_STREAM, 0, unixfd);
	if (_verbose) fprintf (stderr, "* UNIX: initialized socket\n");

	char tmp[512];
	bzero (tmp, 512);

	switch (pid = fork()) {
	case 0:
		/* children */
		close (unixfd[0]);
		write (unixfd[1], "hello", 6);

		close (unixfd[1]);
		exit (EXIT_SUCCESS);
		break;
	default:
		/* parent */
		fprintf (stderr, "* Fork: child %d\n", pid);
		read (unixfd[0], tmp, 512);
		printf ("%s", tmp);
	}

	return _ret_tot;
}
/* ============================================================================ */

void *
Malloc (size_t size) /* wrapper for malloc(3) */
{
	void *_ptr;
	if ((_ptr = malloc (size)) == NULL) {
		perror ("malloc");
		exit (EXIT_FAILURE);
	}
	return _ptr;
}
