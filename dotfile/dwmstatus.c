/* Copyright (C) 2016 Lumin <cdluminate@gmail.com>
 * compile: gcc -Wall -o dwmstatus dwmstatus.c -O2 -lX11
 * MIT License
 */

/* XXX: Linux-only Software
 * Refernce:
 *   1. http://dwm.suckless.org/dwmstatus/
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <assert.h>

#include <sys/utsname.h>
#include <sys/sysinfo.h>

#include <X11/Xlib.h>

#define MAXSTR  512
#define VERSION "0.1"
//#define TEST // gcc -DTEST to compile test binary

/* helper moudle primitives */
static char * module_date(void);
static char * module_sysinfo(void);
static char * module_uname(void);
static char * module_split(void);
static char * module_space(void);
static char * module_cpu(void);


/* <config> dwmstatus :: content */
#define M(name) module_##name
static char * (*status_modules[])(void) = {
	// a set of "static const char *" functions that controls content
	M(uname),
	M(split),
	M(sysinfo),
	M(split),
	M(date),
};

/* <helper> modules that returns simple string */
#define MODULE_STR(name, str) module_##name(void) { \
	return (char*)((str)); \
}
static char * MODULE_STR(space, " ");
static char * MODULE_STR(split, " | ");

/* <helper> get bar */
static const char *
getBar(int percent)
{
	static const char *s[] = {
		"_", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "█"};
	assert(percent <= 100);
	return s[(int)((percent-1)/10)];
}

/* <helper> module output collector */
void
module_collect (char * overview,
		char *(*status_modules[])(void),
		size_t sz_modules)
{
	char *cursor = overview;
	int ret = 0;
	int left = MAXSTR-1;
	int i;
	for(i = 0; i < sz_modules; i++ )
	{
		ret = snprintf(cursor, left, "%s", status_modules[i]() );
		cursor += ret;
		left -= ret;
	}
	return;
}

/* <helper> date */
static char *
module_date (void) {
	static char pc_date[MAXSTR];
	time_t now = time(0);
	strftime(pc_date, MAXSTR, "%Y-%m-%d %H:%M", localtime(&now));
	//strftime(date, MAXSTR, "%Y-%m-%d %H:%M", gmtime(&now)); // UTC
	return pc_date;
}

/* <helper> sysinfo incl. RAM status in Megabytes */
static char *
module_sysinfo (void) {
	static char pc_sysinfo[MAXSTR];
	struct sysinfo s;
	sysinfo(&s);
	// uptime(H) free(M)/all(M) sw(M)
	snprintf(pc_sysinfo, sizeof(pc_sysinfo),
		"UP %.1fH, RAM %dM/%dM, SW %dM",
		((float)s.uptime/3600.),
		(int)((s.totalram-s.freeram)/1048576),
		(int)(s.totalram/1048576),
		(int)((s.totalswap-s.freeswap)/1048576)
		);
	return pc_sysinfo;
}

/* <helper> get uname */
static char *
module_uname (void) {
	static char pc_uname[MAXSTR];
	struct utsname u;
	if(uname(&u)){
		perror("uname failed");
		exit(EXIT_FAILURE);
	}
	snprintf(pc_uname,sizeof(pc_uname),"%s %s",u.sysname,u.nodename);
	return pc_uname;
}

/* <helper,linux-only> get cpu usage via /proc */
static char*
module_cpu (void) {
	//FILE* pf_procstat = fopen("/proc/stat", "r");
	FILE* pf_procstat = NULL;
	unsigned long cpus[4] = {0,0,0,0}; // user,nice,sys,idle
	unsigned long cpue[4] = {0,0,0,0};
	static char pc_cpu[MAXSTR];
#define getProcStatCPU(pf, buf) do {\
	pf = fopen("/proc/stat", "r"); \
	fseek(pf, 0, SEEK_SET); \
	fscanf(pf, "cpu %ld %ld %ld %ld", \
			buf+0, buf+1, buf+2, buf+3); \
	fclose(pf); \
} while(0)
	#define CPUOccupy_(x) ((cpu##x[0] + cpu##x[1] + cpu##x[2]))
	#define CPUTotal_(x) ((cpu##x[0] + cpu##x[1] + cpu##x[2] + cpu##x[3]))
	getProcStatCPU(pf_procstat, cpus);
	usleep(250000);
	getProcStatCPU(pf_procstat, cpue);
	double cpuusage = (double)(CPUOccupy_(e) - CPUOccupy_(s)) * 100. /
		(double)(CPUTotal_(e) - CPUTotal_(s));
	//snprintf(pc_cpu, sizeof(pc_cpu), "CPU %.1f%%", cpuusage); // numerical
	snprintf(pc_cpu, sizeof(pc_cpu), "CPU %.0f%% %s",
			cpuusage, getBar((int)cpuusage)); // num+bar
	return pc_cpu;
}

/* <helper> xsetroot -name xxx */
static void XSetRoot(const char *name){
	Display *display;
	if (( display = XOpenDisplay(0x0)) == NULL ) {
		fprintf(stderr, "cannot open display\n");
		exit(1);
	}
	XStoreName(display, DefaultRootWindow(display), name);
	XSync(display, 0);
	XCloseDisplay(display);
	return;
}

/* dwmstatus :: main */
int
main (int argc, char **argv, char **envp)
{
#if defined(TEST)
	#define MODTEST(fun) do { \
		fprintf(stderr, "=> %s\n", module_##fun()); \
	} while(0)
	MODTEST(uname);
	MODTEST(date);
	MODTEST(sysinfo);
	MODTEST(split);
	MODTEST(space);
	MODTEST(cpu);
#else // TEST
	char pc_overview[MAXSTR];
	module_collect(pc_overview, status_modules,
		sizeof(status_modules)/sizeof(status_modules[0]) );
	fprintf(stderr, "%s\n", pc_overview);
	XSetRoot(pc_overview);
#endif // TEST
	return 0;
}
