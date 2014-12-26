/* bsdbar.h

   part of Bytefreq
   cdluminate@163.com
*/

int i = 0; /* bar state indicator */

void
BSDbar_init (void)
{
    /* write a padding for the bar */
	Write (2, "[ ] ...%", 8);
	return;
}

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
    _BSDbar_refresh (&i, num);
    return;
}

void
BSDbar_clear (void)
{
    /* clear the padding/bar */
	Write (2, "\b\b\b\b\b\b\b\b        ", 16);
	return;
}
