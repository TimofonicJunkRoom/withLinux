/* bsdbar.h

   part of Bytefreq
   cdluminate@163.com
*/

void
BSDbar_init (void)
{
	Write (2, "[ ] ...%", 8);
	return;
}

void
BSDbar_refresh (int *iptr, int num)
{
	/* refresh BSD-style progress bar */
	static char bar[3];
		bar[0] = '-';
		bar[1] = '\\';
		bar[2] = '/';
	static char bb[8]; /* bar buffer */
		bb[0] = '[';
		bb[1] = ' ';
		bb[2] = ']';
		bb[3] = ' ';
		bb[4] = ' '; //
		bb[5] = ' '; //
		bb[6] = ' '; //
		bb[7] = '%';

	if (*iptr > 2 || *iptr < 0) *iptr = 0;
	Write (2, "\b\b\b\b\b\b\b\b", 8);
	snprintf (bb, 8, "[%c] %3d%%", bar[(*iptr)++], num);
	Write (2, bb, 8);
	return;
}
void
BSDbar_clear (void)
{
	Write (2, "\b\b\b\b\b\b\b\b        ", 16);
	return;
}
