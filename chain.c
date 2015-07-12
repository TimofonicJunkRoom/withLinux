#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

struct CHAIN {
	struct CHAIN * prev;
	char * label;
	struct CHAIN * next;
};

/* Create a node of a chain, which can be a head or middle one */
struct CHAIN *
chain_create (char * label)
{
	struct CHAIN * _cp;
	_cp = malloc (sizeof (struct CHAIN) );
	if (NULL == _cp) {
		perror ("malloc");
		exit (EXIT_FAILURE);
	}
	bzero (_cp, sizeof (struct CHAIN) );
	_cp -> label = label;
	return _cp;
}

/* Initialize a new chain */
struct CHAIN *
chain_init (void)
{
	struct CHAIN * head;
	head = chain_create("HEAD");
	/* fill in init values */
	head -> prev = NULL;
	head -> next = NULL;
	/* chain_init returns the head pointer of a new chain */
	return head;
}

struct CHAIN *
chain_append (struct CHAIN * head, struct CHAIN * tailnew)
{
	/* check if the tailnew is valid */
	if (NULL == tailnew) {
		printf ("E: invalid tailnew\n");
		exit (EXIT_FAILURE);
	}
	/* move to the last node */
	struct CHAIN * _cp;
	_cp = head;
	while (NULL != 	_cp -> next) {
		_cp = _cp -> next;
	}
	/* append the tailnew after the last node */
	tailnew -> next = NULL;
	tailnew -> prev = _cp;
	_cp -> next = tailnew;
	/* chain_append returns * of tailnew */	
	return tailnew;
}

void *
chain_destroy (struct CHAIN * head)
{
	/* check if the head is valid */
	if (NULL == head) {
		printf ("E: invalid tailnew\n");
		exit (EXIT_FAILURE);
	}
	/* move to the last node */
	struct CHAIN * _cp;
	_cp = head;
	while (NULL != 	_cp -> next) {
		_cp = _cp -> next;
	}
	/* free until NULL == prev */
	struct CHAIN * _prev;
	do {
		_prev = _cp -> prev;
		free (_cp);
		_cp = _prev;
	} while (NULL != _cp);
	return NULL;
}
