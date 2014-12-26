/* crunch.c
 * data cruncher for Bytefreq, this is a part of bytefreq

   Count Byte/Char freqency, using Serial/Parallel Approaches.
 
   C.D.Luminate <cdluminate AT 163 DOT com> 
   MIT Licence, 2014
 */


 /* TODO : find a proper buffer size
 */
// TODO : use mmap to optimize crunch_serial



/* end use of crunch_unixsock */

/* 131072 Bytes, 128KB buffer */
#define BF_BFSZ_SERI (524288*sizeof(char))
#define BF_BFSZ_PARA (4194304*sizeof(char))
#define BF_BFSZ_UNIX (1048576*sizeof(char))

/* interface */
long crunch_serial (int _fd, long _counter[256], int _verbose);
long crunch_parallel (int _fd, long _counter[256], int _verbose);
long crunch_unixsock (int _fd, long _counter[256], int _verbose);

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
	int _loop;
	long _readn;
	if (_verbose) BSDbar_init ();
	while ((_readn = Read(_fd, _buf, BF_BFSZ_SERI)) > 0) {
		if (_verbose) {
			BSDbar_refresh ((int)(1.0+100.0*((float)_total_read/st.st_size)));
		}
		_total_read += _readn;
		/* #pragma omp parallel for */
		for (_loop = 0; _loop < _readn; _loop++) {
			_counter[(unsigned char)*(_buf+_loop)]++;
		}
	}
	if (_verbose) {
		BSDbar_clear ();
		Write (2, "\n", 2);
	}
	/* free buffer and return */
	free (_buf);
	return _total_read;
}

long crunch_parallel (int _fd, long _counter[256], int _verbose)
{
	/* the value to return */
	long _total_read = 0;

	/* stat */
	struct stat st;
	bzero (&st, sizeof(st));
	fstat (_fd, &st);

	/* flush counter */
	bzero (_counter, 256*sizeof(long));

	/* allocate buffer and flush it */
	char *_buf = (char *)Malloc (BF_BFSZ_PARA);
	bzero (_buf, BF_BFSZ_PARA);

	/* start crunching */
	if (_verbose) BSDbar_init ();
	int _loop;
	long _readn;
	while ((_readn = Read(_fd, _buf, BF_BFSZ_PARA)) > 0) {
		if (_verbose) BSDbar_refresh (100*_total_read/st.st_size);
		_total_read += _readn;
		#pragma omp parallel for
		for (_loop = 0; _loop < _readn; _loop++) {
			_counter[(unsigned char)*(_buf+_loop)]++;
		}
	}
	if (_verbose) {
		BSDbar_clear();
		Write (2, "\n", 1);
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
		Close (unixfd[0]);
		off_t _offset = 0;
		size_t _count = (size_t)st.st_size;
        ssize_t ssize = 0;

		if (_verbose>1) fprintf (stderr, "* Child: start sendfile() to parent.\n");
		while ((ssize = Sendfile (unixfd[1], _fd, &_offset, _count)) > 0) {
            _count -= ssize;
        }
		/* done sendfile(), quit */
		Close (unixfd[1]);
		if (_verbose>1) fprintf (stderr, "* Child: sendfile() finished, exit.\n");
		exit (EXIT_SUCCESS);
	}
		/* parent's matter:
		   read from socket, and count */
	Close (unixfd[1]);
	if (_verbose>1) fprintf (stderr, "* Forked child %d is trying its best running sendfile()...\n", pid);
	char *_buf = (char *) Malloc (BF_BFSZ_UNIX);
	bzero (_buf, BF_BFSZ_UNIX);

	/* start to count */
	int _readn;
	int _loop;
	if (_verbose) BSDbar_init();
	while ((_readn = Read(unixfd[0], _buf, BF_BFSZ_UNIX)) > 0) {
		if (_verbose) {
			BSDbar_refresh ((int)(1.0+100.0*((float)_ret_tot/st.st_size)));
		}
		_ret_tot += _readn;
		for (_loop = 0; _loop < _readn; _loop++) {
			_counter[(unsigned char)*(_buf+_loop)]++;
		}
	}
	if (_verbose) {
		BSDbar_clear ();
		Write (2, "\n", 1);
	}
	/* free buffer and return */
	free (_buf);
	Close (unixfd[0]);
	return _ret_tot;
}
/* ============================================================================ */

