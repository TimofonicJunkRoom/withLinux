/* Copyright (C) 2016 Lumin <cdluminate@gmail.com>
 *
 * XXX: Linux-only Software
 * compile: gcc -Wall -o dwmstatus dwmstatus.c -O2 -lX11
 * install: $HOME/bin
 * xinitrc: while true; do $HOME/bin/dwmstatus; done &
 * customize: modify `static char * (*status_modules[])(void) = ...`
 *
 * MIT License
 *
 * Refernce:
 *   1. http://dwm.suckless.org/dwmstatus/
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
#define VERSION "3"
//#define TEST // gcc -DTEST to compile test binary

/* oops this looks dirty but we need it */
#if defined(__T430s__)
	#define SYSBAT0 "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/PNP0A08:00/device:08/PNP0C09:00/PNP0C0A:00/power_supply/BAT0" // find /sys | ack BAT0
	#define SYSHWMON0 "/sys/devices/virtual/hwmon/hwmon0" // find /sys | ack hwmon
	#define SYSBRIGHT "/sys/class/backlight/intel_backlight/brightness"
	#define SYSBRIGHTMAX "/sys/class/backlight/intel_backlight/max_brightness"
#else
	#error  "Please define your SYSBAT0, SYSHWMON0, SYSBRIGHT!"
#endif

/* set this to 1 to noop *sleep() calls */
static int if_nosleep = 0; // int flag _ no sleep

/* helper moudle primitives */
static char * module_date(void);
static char * module_sysinfo(void);
static char * module_uname(void);
static char * module_cpu(void);
static char * module_battery(void);
static char * module_temperature(void);
static char * module_netupdown(void);
static char * module_audiovolume(void);
static char * module_monbrightness(void);
//
static char * module_split(void);
static char * module_space(void);

/* <config> dwmstatus :: content */
#define M(name) module_##name
static char * (*status_modules[])(void) = {
	// a set of "static const char *" functions that controls content
	M(uname), M(space),
	M(netupdown), M(space),
	M(cpu), M(space),
	M(temperature), M(space),
	M(sysinfo), M(space),
	M(battery), M(space),
	M(audiovolume), M(space),
	M(monbrightness), M(space),
	M(date)
};

#define MODULE_STR(name, str) module_##name(void) { \
	return (char*)((str)); \
}
#define readstuff(path, pattern, dest) do { \
	FILE* pf = fopen(path, "r"); \
	fseek(pf, 0, SEEK_SET); \
	fscanf(pf, pattern, dest); \
} while(0)

/* <helper> modules that returns simple string */
static char * MODULE_STR(space, " ")
static char * MODULE_STR(split, " | ")

/* <helper> get bar */
static const char *
getBar(int percent)
{
	if (percent < 0 || percent > 100)
		return "|?|";
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

/* <helper> monitor brightness */
static char *
module_monbrightness (void)
{
	long br_cur = 0;
	long br_max = 0;
	static char pc_br[MAXSTR];
	{ readstuff(SYSBRIGHT, "%ld", &br_cur); }
	{ readstuff(SYSBRIGHTMAX, "%ld", &br_max); }
	snprintf(pc_br, sizeof(pc_br), "⛭%s", getBar((int)br_cur*100/br_max));
	return pc_br;
}

/* <helper,linux-only ALSA> get audio volume (master gain) */
static char *
module_audiovolume (void)
{
	/* FIXME:BUG: wrong number when the master gain goes to 100% */
	#define CMDGAIN "amixer get Master |" \
	" gawk \"BEGIN{gain=0};NF==6&&/Front (Left|Right)/{gain+=substr(\\$5,2,2)};END{print gain/2}\""
	#define CMDMUTESTATE "amixer sget Master | grep '\\[off\\]' >/dev/null && echo 1 || echo 0"
	#define getMasterGain(pf, buf) do { \
		pf = popen(CMDGAIN, "r"); \
		fscanf(pf, "%d", buf+0); \
		pclose(pf); \
	} while(0)
	#define getMuteState(pf, buf) do { \
		pf = popen(CMDMUTESTATE, "r"); \
		fscanf(pf, "%d", buf+0); \
		pclose(pf); \
	} while(0)

	FILE* pf_avolume = NULL;
	FILE* pf_avmute = NULL;
	static char pc_av[MAXSTR];
	int mutestate = 0;
	int master_gain = 0;
	getMasterGain(pf_avolume, &master_gain);
	getMuteState(pf_avmute, &mutestate);
	if (mutestate) {
snprintf(pc_av, sizeof(pc_av), "♫%s%s", "[M]", getBar(master_gain));
	} else {
snprintf(pc_av, sizeof(pc_av), "♫%d%s", master_gain, getBar(master_gain));
	}
	return pc_av;
}

/* <helper,linux-only> total network up/down */
static char *
module_netupdown (void)
{
#define CMDNET "mawk '" \
	"BEGIN{inbyte=0;outbyte=0};" \
	"NR>2 && $1!~/lo:/ {inbyte+=$2; outbyte+=$10};" \
	"END{print inbyte, outbyte};' /proc/net/dev"
#define getNetUpDown(pf, buf) do { \
	pf = popen(CMDNET, "r"); \
	fscanf(pf, "%lu %lu", buf+0, buf+1); \
	pclose(pf); \
	} while(0)
#define netG (1024*1024*1024)
#define netM (1024*1024)
#define netK (1024)
#define netCompat(count, buf, bufsz) ( \
		(count > netG) ? (snprintf(buf, bufsz, "%.1fGB/s", count/(double)netG)) \
		: (count > netM) ? (snprintf(buf, bufsz, "%.1fMB/s", count/(double)netM)) \
		: (count > netK) ? (snprintf(buf, bufsz, "%.1fKB/s", count/(double)netK)) \
		: (snprintf(buf, bufsz, "%.0fB/s", count)) )

	FILE* pf_netupdown = NULL;
	unsigned long nets[2] = {0,0}; // start up/down
	unsigned long nete[2] = {0,0}; // end up/down
	char pc_down[MAXSTR];
	char pc_up[MAXSTR];
	static char pc_net[MAXSTR];
	getNetUpDown(pf_netupdown, nets);
	usleep(if_nosleep ? 100 : 1000000);
	getNetUpDown(pf_netupdown, nete);
	unsigned long down = nete[0] - nets[0];
	unsigned long up = nete[1] - nets[1];
	//printf("%lu %lu\n", down, up);
	netCompat((double)down, pc_down, sizeof(pc_down));
	netCompat((double)up,   pc_up,   sizeof(pc_up));
	snprintf(pc_net, sizeof(pc_net), "↑%s ↓%s", pc_up, pc_down);
	return pc_net;
}

/* <helper,linux-only> read temperature */
static char *
module_temperature (void)
{
	static char pc_temp0[MAXSTR];
	double temp0;
	readstuff(SYSHWMON0"/temp1_input", "%lf", &temp0);
	snprintf(pc_temp0, sizeof(pc_temp0), "❄%.0f°C", temp0/1000);
	return pc_temp0;
}

/* <helper> battery */
static char *
module_battery (void)
{
#define STREQ(s1,s2) (0==strcmp(s1,s2))
	static char pc_batstatus[MAXSTR];
	static char pc_batcapacity[MAXSTR];
	static char pc_bat[MAXSTR];
	readstuff(SYSBAT0"/status", "%s", pc_batstatus);
	readstuff(SYSBAT0"/capacity", "%s", pc_batcapacity);
	if STREQ("Charging", pc_batstatus) {
		snprintf(pc_bat, sizeof(pc_bat), "⚡%s%%%s%s",
			pc_batcapacity, getBar(atoi(pc_batcapacity)), "[+]");
	} else if STREQ("Discharging", pc_batstatus) {
		snprintf(pc_bat, sizeof(pc_bat), "⚡%s%%%s%s",
			pc_batcapacity, getBar(atoi(pc_batcapacity)), "[-]");
	} else if STREQ("Unknown", pc_batstatus) {
		snprintf(pc_bat, sizeof(pc_bat), "⚡%s%%%s%s",
			pc_batcapacity, getBar(atoi(pc_batcapacity)), "[A/C]");
	} else {
		snprintf(pc_bat, sizeof(pc_bat), "⚡%s%%%s[%s]",
			pc_batcapacity, getBar(atoi(pc_batcapacity)), pc_batstatus);
	}
	return pc_bat;
}

/* <helper> date */
static char *
module_date (void) {
	static char pc_date[MAXSTR];
	time_t now = time(0);
	strftime(pc_date, MAXSTR, "⛅ %Y-%m-%d %H:%M:%S", localtime(&now));
	//strftime(date, MAXSTR, "%Y-%m-%d %H:%M", gmtime(&now)); // UTC
	return pc_date;
}

/* <helper> sysinfo incl. RAM status in Megabytes */
static char *
module_sysinfo (void) {
	static char pc_sysinfo[MAXSTR];
	struct sysinfo s;
	sysinfo(&s);
	snprintf(pc_sysinfo, sizeof(pc_sysinfo),
		"☕%.1fh ♻%.0f%%%s",
		((float)s.uptime/3600.),
		100.*(float)(s.totalram-s.freeram)/(float)s.totalram,
		getBar((int)(100.*(float)(s.totalram-s.freeram)/(float)s.totalram)));
	//	"UP %.1fH, RAM %dM/%dM, SW %dM",
	//	((float)s.uptime/3600.),
	//	(int)((s.totalram-s.freeram)/1048576),
	//	(int)(s.totalram/1048576),
	//	(int)((s.totalswap-s.freeswap)/1048576)
	//	);
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
	snprintf(pc_uname, sizeof(pc_uname), "⚛%s", u.nodename);
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
	usleep(if_nosleep ? 100 : 250000);
	getProcStatCPU(pf_procstat, cpue);
	double cpuusage = (double)(CPUOccupy_(e) - CPUOccupy_(s)) * 100. /
		(double)(CPUTotal_(e) - CPUTotal_(s));
	//snprintf(pc_cpu, sizeof(pc_cpu), "CPU %.1f%%", cpuusage); // numerical
	snprintf(pc_cpu, sizeof(pc_cpu), "♥%.0f%%%s",
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
	MODTEST(battery);
	MODTEST(temperature);
	MODTEST(netupdown);
	MODTEST(audiovolume);
	MODTEST(monbrightness);
#else // TEST
	if (argc > 1) {
		if_nosleep = 1; // toggle fast status scan
	}
	char pc_overview[MAXSTR];
	module_collect(pc_overview, status_modules,
		sizeof(status_modules)/sizeof(status_modules[0]) );
	//fprintf(stderr, "%s\n", pc_overview);
	XSetRoot(pc_overview);
#endif // TEST
	return 0;
}
