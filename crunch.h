/* crunch.c
 * data cruncher for Bytefreq, this is a part of bytefreq

   Count Byte/Char freqency, using Serial/Parallel Approaches.
 
   C.D.Luminate <cdluminate AT 163 DOT com> 
   MIT Licence, 2014
 */


/* TODO : malloc error handle
 * TODO : find a proper buffer size
 */

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

/* 131072 Bytes, 128KB buffer */
#define BF_BFSZ_SERI (131072*sizeof(char))
#define BF_BFSZ_PARA (524288*sizeof(char))

long crunch_serial (int _fd, long _counter[256], int _verbose);
long crunch_parallel (int _fd, long _counter[256], int _verbose);


long crunch_serial (int _fd, long _counter[256], int _verbose)
{
	/* the value to return */
	long _total_read = 0;

	/* flush counter */
	bzero (_counter, 256*sizeof(long));

	/* allocate buffer and flush it */
	char *_buf;
	_buf = (char *)malloc (BF_BFSZ_SERI);
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
	_buf = (char *)malloc (BF_BFSZ_PARA);
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
