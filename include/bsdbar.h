/* bsdbar.h

   part of Bytefreq
   cdluminate@163.com
*/

/* SYNOPSIS

   0. #include "bsdbar.h"
   
   1. when going to start the progress bar, invoke
      BSDbar_init (void);

   2. when going to refresh the progress bar, invoke
      BSDbar_refresh (int proportion);

   3. when going to clear the bar, invoke
      BSDbar_clear (void);

   4. that's all
*/

/* INTERFACE */
void BSDbar_init (void);
void BSDbar_clear (void);
void BSDbar_refresh (int num);
/* END INTERFACE */

int _bsdbar_indicator = 0; /* bar state indicator, internal use */
struct _bsdbar {
	char bar;
	struct _bsdbar * next;
} bar1, bar2, bar3;

struct _bsdbar * _bar_cursor = &bar1;

void
BSDbar_init (void)
{
    /* write a padding for the bar */
	Write (2, "[ ] ...%", 8);
	/* build a chain cycle */
	bar1.bar = '-';
	bar2.bar = '\\';
	bar3.bar = '/';
	bar1.next = &bar2;
	bar2.next = &bar3;
	bar3.next = &bar1;

	return;
}

/* this function is for internal use */
void
_BSDbar_refresh (char _bar, int num)
{
	/* refresh BSD-style progress bar */
    /* whole buffer of the bar */
	static char bb[8] = {
        '[', ' ', ']', ' ', ' ', ' ', ' ', '%'
    };
	Write (2, "\b\b\b\b\b\b\b\b", 8); /* clear the previous bar */
	snprintf (bb, 8, "[%c] %3d%%", _bar, num); /* prepare buffer */
	Write (2, bb, 8); /* print the buffer to stderr */
	return;
}

void
BSDbar_refresh (int num)
{
    /* note that 'int num' is the proportion to display */
    _BSDbar_refresh (_bar_cursor -> bar, num);
	_bar_cursor = _bar_cursor -> next;
    return;
}

void
BSDbar_clear (void)
{
    /* clear the padding/bar and newline*/
	Write (2, "\b\b\b\b\b\b\b\b        \n", 17);
	return;
}
