/* fir.c

   A five-in-a-row chess game, 
   which is text-based.

   C.D.Luminate <cdluminate@163.com>
   2014/04/18
 */

#include <stdio.h>
#include <stdlib.h>
#include "fir_flags.h"

char geom[15][15];
int size = 15;

void 
pmatrix (char g[15][15])  // print the matrix of game board
{
	int i,j;
	printf ("  1 2 3 4 5 6 7 8 9 A B C D E F \n");
	for (j= size-1; j>=0; j--) {
		putchar (' ');
		for (i=0; i < size; i++) {
			putchar (' ');
			putchar (g[j][i]);
		}
		printf ("  %d", j+1);
		putchar ('\n');
	}
}

int
settle (char *target, char token)  // settle a chess on game board
{
	if (*target == '.') {
		*target = token;
		return 1; //true
	}
	else
		return 0; //false
}

void getvalidstep (int *stepx, int *stepy, char g[15][15]);

int
main (void)
{
	int i,j;
	/* initialize board with . */
	for (j=0; j<15; j++) {
		for (i=0; i<15; i++) {
			geom[j][i] = '.';
		}
	}
	db ("Game board initialized.\n");
	pmatrix (geom); /* print */

	/* game loop */
	int loop = 1;
	while ( 1 ) {
		/* let me know the loop */
		printf ("DB : current loop %d\n", loop);

		/* read a step */
		int x,y;
		getvalidstep (&x, &y, geom);
		db ("get step");
		
		//settle ( &geom[y][x], 'X');
		settle ( &geom[y][x], (loop%2==1)?('X'):('O') );
		pmatrix (geom);


		/* if flag-win */
		if ( flagwin (geom, 'X') ) {
			db ("X win");
			exit (0);
		}
		if ( flagwin (geom, 'O') ) {
			db ("O win");
			exit (0);
		}
		loop++;
	}


	return 0;
}

void 
getvalidstep (int *stepx, int *stepy, char g[15][15])
{
	db ("enter function getvalidstep");
	/* input string into it */
	//char instr[7] = {'\0'};
	//fgets (instr, 7, stdin);

	/* read the location you refered */
	int lx, ly;
	db ("format must be \"%d %d\" ");
	scanf ("%d %d", &lx, &ly);
	*stepy = ly-1;
	*stepx = lx-1;

	/* if go beyond */
	if (
		*stepx < 0 || *stepx >= size ||
		*stepy < 0 || *stepy >= size
	   ) {
		db ("step out of range");
		getvalidstep (stepx, stepy, g);
	}

	/* inspact if target location is avaliable */
	if ( g[*stepy][*stepx] != '.' ) {
		db ("occupied");
		getvalidstep (stepx, stepy, g);
	}
}

