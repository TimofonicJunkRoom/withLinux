#include <unistd.h>
#include "cdalog.h"

void
train (void)
{
	int i = 0;
	while (i <= 3) {
		LOG_INFOF ("iteration %d\n", i);
		i++;
		LOG_DEBUG ("sleeping for 1 sec\n");
		sleep (1);
	}
	return;
}

int
main (void)
{
	LOG_INFO ("hello, this is a demo of libcdalog\n");
	LOG_WARN ("example of warning\n");
	LOG_ERROR ("example of error\n");
	LOG_DEBUG ("example of debug information\n");
	
	train ();

	LOG_INFOF ("%s\n", "Happy hacking!");
	return 0;
}
