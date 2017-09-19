/*
 * http://invisible-island.net/ncurses/ncurses-intro.html
 */

#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <string.h>

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

	// https://stackoverflow.com/questions/19614156/c-curses-remove-blinking-cursor-from-game
	curs_set(0); // no blinking cursor;

	if (has_colors()) {
		start_color();
		init_pair(1, COLOR_GREEN, COLOR_BLACK);
		init_pair(2, COLOR_RED,   COLOR_BLACK);
		init_pair(3, COLOR_YELLOW,COLOR_BLACK);
	}

	char buf[] = "                            ";

	// moving banners
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

	// progress hinter
	for (int i = 0; i <=100; i++) {
		clear();
		attrset(COLOR_PAIR(3) | A_BOLD);
		snprintf(buf, sizeof(buf), "-> Progress %.2f%%", i/100.);
		mvaddstr(20, 5, buf);
		refresh();
		usleep(1000*10);
	}

	// rolling bar
	char* banner = (char*)malloc((COLS+1)*sizeof(char));
	bzero(banner, (COLS+1)*sizeof(char));
	int size_w = (int)((1-0.618)*COLS);
	clear();
	for (int i = 0; i <= 300; i++) {
		clear();
		memset(banner, '>', COLS*sizeof(char));
		// mask the sliding window with white spaces
		for (int j = 0; j < size_w; j++) {
			*(banner + (i+j)%COLS) = ' ';
		}
		attrset(COLOR_PAIR(1) | A_BOLD);
		mvaddstr((int)(.5*LINES), 0, banner);
		//printf("%s\n", banner);
		refresh();
		usleep(1000*10);
	}
	free(banner);
	clear();

	// msg box ?
#define MSG "To be or not to be, that is a question."
	mvaddstr((int)(.5*LINES), (int)(.5*COLS)-(int)(.5*sizeof(MSG)), MSG);
	for (int j = 0; j < sizeof(MSG)+3; j++) {
		int yoff = (int)(.5*LINES);
		int xoff = (int)(.5*COLS)-(int)(.5*sizeof(MSG));
		if (0==j || sizeof(MSG)+2==j) {
			mvaddch(yoff-2, xoff-2+j, '.');
			mvaddch(yoff+2, xoff-2+j, '.');
			mvaddch(yoff-1, xoff-2+j, '|');
			mvaddch(yoff+1, xoff-2+j, '|');
			mvaddch(yoff+0, xoff-2+j, '|');
		} else {
			mvaddch(yoff-2, xoff-2+j, '-');
			mvaddch(yoff+2, xoff-2+j, '-');
		}
	}
	refresh();
	usleep(1000*3000);

	finish(0);
	return 0;
}
