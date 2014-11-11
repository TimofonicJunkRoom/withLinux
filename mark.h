/* mark.h
 * data cruncher for Bytefreq, this is a part of bytefreq

   Count Byte/Char freqency, using Serial/Parallel Approaches.

   C.D.Luminate <cdluminate AT 163 DOT com> 
   MIT Licence, 2014
 */

int mark_control (int _mark[256]);

int mark_symbol (int _mark[256]);

int mark_number (int _mark[256]);

int mark_upper (int _mark[256]);

int mark_lower (int _mark[256]);

int
_count_marker (int _type, int _mark[256])
{
/* these definitions only used in following part of this file */
#define CONTROL 1
#define SYMBOL  2
#define NUMBER  4
#define UPPER   8
#define LOWER   16
#define ALPHA   32
#define ALL     256
#define eqlo(j) (_type == (j)) /* equal to _type? */
/* end def */
	int _lo; /* loop */
	if (_type == ALL) {
		/* then set all marks as 1 and return */
		for (_lo = 0; _lo < 256; _lo++) {
			_mark[_lo] = 1;
		}
		return 256;
	}
	/* not all */
	for (_lo = 0; _lo < 256; _lo++) {
		switch (_lo) {
		/* the switch{} block used gcc extension */
		case 0 ... 31:
			if (eqlo(CONTROL)) _mark[_lo] = 1;
			break;
		case 32 ... 47:
			if (eqlo(SYMBOL)) _mark[_lo] = 1;
			break;
		case 48 ... 57:
			if (eqlo(NUMBER)) _mark[_lo] = 1;
			break;
		case 58 ... 64:
			if (eqlo(SYMBOL)) _mark[_lo] = 1;
			break;
		case 65 ... 90:
			if (eqlo(UPPER) || eqlo(ALPHA)) _mark[_lo] = 1;
			break;
		case 91 ... 96:
			if (eqlo(SYMBOL)) _mark[_lo] = 1;
			break;
		case 97 ... 122:
			if (eqlo(LOWER) || eqlo(ALPHA)) _mark[_lo] = 1;
			break;
		case 123 ... 126:
			if (eqlo(SYMBOL)) _mark[_lo] = 1;
			break;
		case 127:
			if (eqlo(CONTROL)) _mark[_lo] = 1;
			break;
		default:
			;
		}
	}
	return 0;
}
