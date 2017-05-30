/* setuid version of brightness helper */
// personal backlight helper for T430s, setuid version
// Copyright © 2017 Lumin
//
// Usage: 1. sudo chown root:root <ELF>
//        2. sudo chmod u+s <ELF>
//        3. <ELF>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// configuration
#define BLDIR "/sys/class/backlight/intel_backlight"
#define BL_MAX BLDIR"/max_brightness"
#define BL_CUR BLDIR"/brightness"

#define readstuff(pf, path, pattern, buf) do { \
	pf = fopen(path, "r"); \
	fscanf(pf, pattern, buf); \
	fclose(pf); \
} while(0)
#define writestuff(stuff, path, pattern, buf) do { \
	pf = fopen(path, "w"); \
	fprintf(pf, pattern, buf); \
	fclose(pf); \
} while(0)

/* <helper> */
long
getBrightnessMax (void) {
	long l_bmax = 0;
	FILE* pf = NULL;
	readstuff(pf, BL_MAX, "%ld", &l_bmax);
	fprintf(stderr, "=> l_bmax %ld\n", l_bmax);
	return l_bmax;
}
#define getBM getBrightnessMax

/* <helper> */
long
getBrightnessCur (void) {
	long l_bcur = 0;
	FILE* pf = NULL;
	readstuff(pf, BL_CUR, "%ld", &l_bcur);
	fprintf(stderr, "=> l_bcur %ld\n", l_bcur);
	return l_bcur;
}
#define getBC getBrightnessCur

/* <helper> */
long
getSabitizedBrightness (long value) {
	long max = getBM();
	if (value > max)
		return max;
	else if (value < 0)
		return 0;
	else
		return value;
}
#define getSB(x) getSabitizedBrightness(x)

/* <helper> */
long
writeBrightness (long b) {
	FILE* pf = NULL;
	fprintf(stderr, "=> brightness %ld\n", b);
	writestuff(pf, BL_CUR, "%ld", b);
	return b;
}
#define writeBL(x) writeBrightness(x)

/* ************* */
int
main (int argc, char **argv, char **envp)
{
	// arg check
	if (2 != argc)
		exit(EXIT_SUCCESS);
	// adjust brightness
	long bl = 0;
	switch (argv[1][0]) {
	case '+': // +10%
		bl = getSB( getBC() + (long)(getBM()/10.) );
		break;
	case '-': // -10%
		bl = getSB( getBC() - (long)(getBM()/10.) );
		break;
	default: // =argv[1]%
		bl = getSB( (long)(atol(argv[1])/100. * getBM()) );
		break;
	}
	// promotion and write new brightness
	if (0 != geteuid()) {
		perror("You must setuid this program to root.");
		exit(EXIT_FAILURE);
	}
	setuid(geteuid());
	writeBL(bl);
	return 0;
}
