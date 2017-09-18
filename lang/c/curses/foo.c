/*
 * http://invisible-island.net/ncurses/ncurses-intro.html
 */

#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

#include <curses.h> // -lncurses

static void
finish(int sig)
{
	endwin();
	exit(0);
}

int
main(int argc, char *argv[])
{
	(void) signal(SIGINT, finish);

	(void) initscr();
	keypad(stdscr, TRUE);
	(void) nonl();
	(void) cbreak();
	(void) echo();

	if (has_colors()) {
		start_color();
		init_pair(1, COLOR_GREEN, COLOR_BLACK);
		init_pair(2, COLOR_RED,   COLOR_BLACK);
		init_pair(3, COLOR_YELLOW,COLOR_BLACK);
	}

	char buf[] = "                            ";
	for (int i = 0; i < 20; i++) {
		if (i==10) clear();

		move(i, i);
		attrset(COLOR_PAIR(1) | A_BOLD);
		addstr("hello\n");
		//refresh();
		//usleep(1000 * 100);

		attrset(COLOR_PAIR(2) | A_BOLD);
		snprintf(buf, sizeof(buf), "LINES %d COLS %d\n", LINES, COLS);
		mvaddstr(40-i, i, buf);
		refresh();
		usleep(1000 * 100);

		clear();
	}

	for (int i = 0; i <=100; i++) {
		clear();
		attrset(COLOR_PAIR(3) | A_BOLD);
		snprintf(buf, sizeof(buf), "-> Progress %.2f%%", i/100.);
		mvaddstr(20, 5, buf);
		refresh();
		usleep(1000*10);
	}

	finish(0);
	return 0;
}
