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

/* used by crunch_unixsock */
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/sendfile.h>

#include "sock_wrapper.h"
/* end use of crunch_unixsock */

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
	char *_buf = (char *)Malloc (BF_BFSZ_PARA);
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
	/* doesn't read stdin */
	if (_fd == fileno(stdin)) {
		fprintf (stderr, "* Error: crunch_unixsock() doesn't read stdin\n");
		exit (EXIT_SUCCESS);
	}
	/* flush counter */
	bzero (_counter, 256*sizeof(long));

#define UNIXPATH "/tmp/bytefreq_socket_unix"
	/* prepare misc */
	pid_t pid;

	struct stat st;
	fstat (_fd, &st);

	long _ret_tot = 0;

	int unixfd[2];
	bzero (unixfd, sizeof(unixfd));

	/* launch socket */
	Socketpair (AF_UNIX, SOCK_STREAM, 0, unixfd);
	if (_verbose) fprintf (stderr, "* UNIX: initialized socket\n");

	char tmp[512];
	bzero (tmp, 512);

	/* child write unixfd[1], parent read unixfd[0] */
	if ((pid = Fork()) == 0) {
		/* children's matter:
		   just sendfile() */
		close (unixfd[0]);
		if (_verbose) fprintf (stderr, "* Child: start sendfile() to parent.\n");
		sendfile (unixfd[1], _fd, NULL, st.st_size);
		/* done sendfile(), quit */
		close (unixfd[1]);
		if (_verbose) fprintf (stderr, "* Child: done, exit.\n");
		exit (EXIT_SUCCESS);
	}
		/* parent's matter:
		   read from socket, and count */
	close (unixfd[1]);
	if (_verbose) fprintf (stderr, "* Fork: child %d\n", pid);
	char *_buf = (char *) Malloc (BF_BFSZ_UNIX);
	bzero (_buf, BF_BFSZ_UNIX);

	/* start to count */
	int _readn;
	int _loop;
	if (_verbose) write (2, "!", 1);
	while ((_readn = read(unixfd[0], _buf, BF_BFSZ_UNIX)) > 0) {
		if (_verbose) write (2, ".", 1);
		_ret_tot += _readn;
		for (_loop = 0; _loop < _readn; _loop++) {
			_counter[(unsigned char)*(_buf+_loop)]++;
		}
	}
	/* free buffer and return */
	free (_buf);
	if (_verbose) write (2, "!\n", 2);
	close (unixfd[0]);
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
