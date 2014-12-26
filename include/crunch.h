/* crunch.h
 * data cruncher for Bytefreq, this is a part of bytefreq

   Count Byte/Char freqency, using Serial/Parallel Approaches.
 
   C.D.Luminate <cdluminate AT 163 DOT com> 
   MIT Licence, 2014
 */

/* TODO : find a proper buffer size */
// TODO : use mmap to optimize crunch_serial ?

/* 131072 Bytes, 128KB buffer */
#define BF_BFSZ_SERI (524288*sizeof(char))
#define BF_BFSZ_PARA (4194304*sizeof(char))
#define BF_BFSZ_UNIX (1048576*sizeof(char))

/* interface */
long crunch_serial (int _fd, long _counter[256], int _verbose);
long crunch_parallel (int _fd, long _counter[256], int _verbose);
long crunch_unixsock (int _fd, long _counter[256], int _verbose);

/* implementation */
#include "crunch/crunch_serial.h"
#include "crunch/crunch_parallel.h"
#include "crunch/crunch_unix.h"
