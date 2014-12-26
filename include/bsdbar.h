/* bsdbar.h

   part of Bytefreq
   cdluminate@163.com
*/

/* SYNOPSIS

   0. #include "bsdbar.h"
   
   1. when going to start the progress bar, invoke
      BSDbar_init ();

   2. when going to refresh the progress bar, invoke
      BSDbar_refresh (int proportion);

   3. when going to clear the bar, invoke
      BSDbar_clear ();

   4. that's all
*/

/* INTERFACE */
void BSDbar_init (void);
void BSDbar_clear (void);
void BSDbar_refresh (int num);
/* END INTERFACE */

int _bsdbar_indicator = 0; /* bar state indicator, internal use */

void
BSDbar_init (void)
{
    /* write a padding for the bar */
	Write (2, "[ ] ...%", 8);
	return;
}

/* this function is for internal use */
void
_BSDbar_refresh (int *iptr, int num)
{
	/* refresh BSD-style progress bar */

    /* the spinning bar itself */
	static char bar[3] = { '-', '\\', '/' };

    /* whole buffer of the bar */
	static char bb[8] = {
        '[', ' ', ']', ' ', ' ', ' ', ' ', '%'
    };

    /* if bar status is out of range, reset */
	if (*iptr > 2 || *iptr < 0) *iptr = 0;
	Write (2, "\b\b\b\b\b\b\b\b", 8); /* clear the padding */
	snprintf (bb, 8, "[%c] %3d%%", bar[(*iptr)++], num); /* prepare buffer */
	Write (2, bb, 8); /* print the buffer to stderr */
	return;
}

void
BSDbar_refresh (int num)
{
    /* note that 'int num' is the proportion to display */
    _BSDbar_refresh (&_bsdbar_indicator, num);
    return;
}

void
BSDbar_clear (void)
{
    /* clear the padding/bar and newline*/
	Write (2, "\b\b\b\b\b\b\b\b        \n", 17);
	return;
}
