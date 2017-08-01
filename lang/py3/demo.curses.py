#!/usr/bin/python3
# UTF8

import curses
''' curses is shipped in standard python package '''

def main():
  sc = curses.initscr()
  sc.clear()
  sc.border(0)
  sc.addstr(12, 25, "Python curses!")
  sc.refresh()
  key = sc.getch()
  curses.endwin()

main()
