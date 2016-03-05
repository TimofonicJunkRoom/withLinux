#include "cdalog.h"

int
main (void)
{
	LOG_INFO ("hello, this is a demo of libcdalog\n");
	LOG_WARN ("example of warning\n");
	LOG_ERROR ("example of error\n");
	LOG_DEBUG ("example of debug information\n");
	return 0;
}
