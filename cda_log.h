#ifndef CDA_LOG_H_
#define CDA_LOG_H_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/timeb.h>
#include <sys/types.h>

#include "cda.h" /* for the color definitions */

struct timeb timeb_s;
char _cda_logbuf[4096] = {0};

/* backend function */
void
_CDA_LOG_CORE (char level,
           struct timeb * timebp,
           pid_t pid,
           __typeof__(__FILE__) file,
           __typeof__(__LINE__) line,
           char * msgstring) 
{
	ftime (timebp);
	struct tm * ptm = gmtime (&timebp->time);
	fprintf (stderr, (level=='I')?CDA_COLOR_GREEN_B
			:(level=='W')?CDA_COLOR_YELLOW_B
			:(level=='E')?CDA_COLOR_RED_B
			:CDA_COLOR_RESET);
	fprintf (stderr, "%1c%02d%02d %02d:%02d:%02d.%03d %05d %s:%d] %s", level,
		   	ptm->tm_mon, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec,
		   	timebp->millitm, pid, file, line, msgstring);
	fprintf (stderr, CDA_COLOR_RESET);
	return;
}

#define LOG_INFO(_cda_msg) do { \
	_CDA_LOG_CORE ('I', &timeb_s, getpid(), __FILE__, __LINE__, _cda_msg); \
} while (0)

#define LOG_WARN(_cda_msg) do { \
	_CDA_LOG_CORE ('W', &timeb_s, getpid(), __FILE__, __LINE__, _cda_msg); \
} while (0)

#define LOG_ERROR(_cda_msg) do { \
	_CDA_LOG_CORE ('E', &timeb_s, getpid(), __FILE__, __LINE__, _cda_msg); \
} while (0)

#define LOG_INFOF(...) do { \
	snprintf (_cda_logbuf, 4095, ##__VA_ARGS__); \
	LOG_INFO (_cda_logbuf); \
} while (0)

#define LOG_WARNF(...) do { \
	snprintf (_cda_logbuf, 4095, ##__VA_ARGS__); \
	LOG_WARN (_cda_logbuf); \
} while (0)

#define LOG_ERRORF(...) do { \
	snprintf (_cda_logbuf, 4095, ##__VA_ARGS__); \
	LOG_ERROR (_cda_logbuf); \
} while (0)

//
//int
//main (void)
//{
//		struct timeb * tp = (struct timeb *) malloc (sizeof(struct timeb));
//		ftime (tp);
//
//		printf ("%ld\n", time(&tp->time));
//		printf ("%s", ctime(&tp->time));
//		ptm = gmtime (&tp->time);
//		printf ("%02d:%02d:%02d.%03d\n",
//				ptm -> tm_hour,
//				ptm -> tm_min,
//				ptm -> tm_sec,
//				tp -> millitm);
//		printf ("%03d\n", tp->millitm);
//
//		free (tp);
//		return 0;
//}
//

#endif /* CDA_LOG_H_ */
