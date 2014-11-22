/* crunch.c
 * data cruncher for Bytefreq, this is a part of bytefreq

   Count Byte/Char freqency, using Serial/Parallel Approaches.
 
   C.D.Luminate <cdluminate AT 163 DOT com> 
   MIT Licence, 2014
 */


 /* TODO : find a proper buffer size
 */
// TODO : use mmap to optimize crunch_serial

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

/* interface */
long crunch_serial (int _fd, long _counter[256], int _verbose);
long crunch_parallel (int _fd, long _counter[256], int _verbose);
long crunch_unixsock (int _fd, long _counter[256], int _verbose);

/* wrapper */
void *Malloc (size_t size);
ssize_t Sendfile(int out_fd, int in_fd, off_t *offset, size_t count);

/* other */
void BSDbar (int *turn, int num);

/* ============================================================================ */
long crunch_serial (int _fd, long _counter[256], int _verbose)
{
	/* stat */
	struct stat st;
	bzero (&st, sizeof(st));

	fstat (_fd, &st);

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
	int turn = 0; /* bsd bar */
	int _loop;
	long _readn;
	if (_verbose) write (2, "[ ] ...%", 8);
	while ((_readn = read(_fd, _buf, BF_BFSZ_SERI)) > 0) {
		if (_verbose) {
			BSDbar (&turn, (int)(100.0*_total_read/st.st_size));
		}
		_total_read += _readn;
		/* #pragma omp parallel for */
		for (_loop = 0; _loop < _readn; _loop++) {
			_counter[(unsigned char)*(_buf+_loop)]++;
		}
	}
	if (_verbose) write (2, "%\n", 2);
	/* free buffer and return */
	free (_buf);
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
	/* for tht BSD style bar */
	int turn = 0;

	/* doesn't read stdin */
	if (_fd == fileno(stdin)) {
		fprintf (stderr, "* Error: crunch_unixsock() doesn't read stdin\n");
		exit (EXIT_SUCCESS);
	}
	/* flush counter */
	bzero (_counter, 256*sizeof(long));

	/* prepare misc */
	pid_t pid;

	struct stat st;
		fstat (_fd, &st);
	if (_verbose>1) fprintf (stderr, "* debug: file size [%lld]\n", (long long)st.st_size);

	long _ret_tot = 0;

	int unixfd[2];
		bzero (unixfd, sizeof(unixfd));

	/* launch socket */
	Socketpair (AF_UNIX, SOCK_STREAM, 0, unixfd);
	if (_verbose>1) fprintf (stderr, "* UNIX: initialized socket\n");

	/* child write unixfd[1], parent read unixfd[0] */
	if ((pid = Fork()) == 0) {
		/* children's matter:
		   just sendfile() */
		close (unixfd[0]);
		off_t _offset = 0;
		size_t _count = (long long)st.st_size;

		if (_verbose>1) fprintf (stderr, "* Child: start sendfile() to parent.\n");
		while (sendfile (unixfd[1], _fd, &_offset, _count) > 0) { ;}
		/* done sendfile(), quit */
		close (unixfd[1]);
		if (_verbose>1) fprintf (stderr, "* Child: sendfile() finished, exit.\n");
		exit (EXIT_SUCCESS);
	}
		/* parent's matter:
		   read from socket, and count */
	close (unixfd[1]);
	if (_verbose) fprintf (stderr, "* Forked child %d is trying its best running sendfile()...\n", pid);
	char *_buf = (char *) Malloc (BF_BFSZ_UNIX);
	bzero (_buf, BF_BFSZ_UNIX);

	/* start to count */
	int _readn;
	int _loop;
	if (_verbose) write (2, "[ ] ...%", 8);
	while ((_readn = read(unixfd[0], _buf, BF_BFSZ_UNIX)) > 0) {
		if (_verbose) {
			BSDbar (&turn, (int)(100.0*_ret_tot/st.st_size));
		}
		_ret_tot += _readn;
		for (_loop = 0; _loop < _readn; _loop++) {
			_counter[(unsigned char)*(_buf+_loop)]++;
		}
	}
	if (_verbose) write (2, "\n", 1);
	/* free buffer and return */
	free (_buf);
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

ssize_t
Sendfile(int out_fd, int in_fd, off_t *offset, size_t count)
{
	ssize_t _ = sendfile (out_fd, in_fd, offset, count);
	if (_ == -1) {
		perror ("sendfile");
		exit (EXIT_FAILURE);
	}
	return _;
}

void
BSDbar (int *iptr, int num)
{
	/* BSD-style progress bar */
	static char bar[3];
		bar[0] = '-';
		bar[1] = '\\';
		bar[2] = '/';
	static char bb[8]; /* bar buffer */
		bb[0] = '[';
		bb[1] = ' ';
		bb[2] = ']';
		bb[3] = ' ';
		bb[4] = ' '; //
		bb[5] = ' '; //
		bb[6] = ' '; //
		bb[7] = '%';

	if (*iptr > 2 || *iptr < 0) *iptr = 0;
	write (2, "\b\b\b\b\b\b\b\b", 8);
	snprintf (bb, 8, "[%c] %3d%%", bar[(*iptr)++], num);
	write (2, bb, 8);
	return;
}
