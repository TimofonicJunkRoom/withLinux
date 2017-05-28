/* Copyright (C) 2016 Lumin <cdluminate@gmail.com>
 * compile: gcc -Wall -o dwmstatus dwmstatus.c -O2 -lX11
 * MIT License
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdarg.h>

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

/* <config> dwmstatus :: content */
#define M(name) module_##name
static char * (*status_modules[])(void) = {
	// a set of "static const char *" functions that controls content
	M(uname),
	M(sysinfo),
	M(date)
};

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

/* <helper> split line */
static char *
module_split (void) {
	return (char*)" | ";
}

/* <helper> space */
static char *
module_space (void) {
	return (char*)" ";
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
#else // TEST
	char pc_overview[MAXSTR];
	module_collect(pc_overview, status_modules,
		sizeof(status_modules)/sizeof(status_modules[0]) );
	fprintf(stderr, "%s\n", pc_overview);
	XSetRoot(pc_overview);
#endif // TEST
	return 0;
}
