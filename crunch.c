/* crunch.c
 * data cruncher for Bytefreq 
 * cdluminate
 */

/* TODO : malloc error handle
 *
 */

#include "crunch.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

#define BF_BFSZ_SERI 32768
#define BF_BFSZ_PARA 131072

long crunch_serial (int _fd, long _counter[256])
{
	/* the value to return */
	long _total_read = 0;

	/* flush counter */
	bzero (_counter, 256 * sizeof(long));

	/* allocate buffer and flush it */
	char *_buf;
	_buf = (char *)malloc (BF_BFSZ_SERI);
	bzero (_buf, BF_BFSZ_SERI);

	/* start crunching */
	int _loop;
	long _readn;
	while ((_readn = read(_fd, _buf, BF_BFSZ_SERI)) > 0) {
		/* #pragma omp parallel for */
		for (_loop = 0; _loop < _readn; _loop++) {
			_counter[(unsigned int)*(_buf+_loop)]++;
			_total_read +=  _readn;
		}
	}
	/* free buffer and return */
	free (_buf);
	return _total_read;
}

long crunch_parallel (int _fd, long _counter[256])
{
	/* the value to return */
	long _total_read = 0;

	/* flush counter */
	bzero (_counter, 256 * sizeof(long));

	/* allocate buffer and flush it */
	char *_buf;
	_buf = (char *)malloc (BF_BFSZ_PARA);
	bzero (_buf, BF_BFSZ_PARA);

	/* start crunching */
	int _loop;
	long _readn;
	while ((_readn = read(_fd, _buf, BF_BFSZ_PARA)) > 0) {
		/* #pragma omp parallel for */
		for (_loop = 0; _loop < _readn; _loop++) {
			_counter[(unsigned int)*(_buf+_loop)]++;
			_total_read +=  _readn;
		}
	}
	/* free buffer and return */
	free (_buf);
	return _total_read;
}
