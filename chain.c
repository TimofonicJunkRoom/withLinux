/* libstack
   2015 Lumin
   BSD-2-Clause
 */
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

struct CHAIN {
	struct CHAIN * prev;
	int    id;
	char * label;
	void * blob; /* extentions */
	struct CHAIN * next;
};

struct CHAIN *
_chain_tail (struct CHAIN * head)
{
	/* check if head is valid */
	if (NULL == head) {
		printf ("E: _chain_tail(): NULL == head\n");
		exit (EXIT_FAILURE);
	}
	/* move to the last node */
	struct CHAIN * _cp;
	_cp = head;
	while (NULL != 	_cp -> next) {
		_cp = _cp -> next;
	}
	return _cp;
}

/* Create a node of a chain, which can be a head or middle one */
struct CHAIN *
chain_create (int id, char * label, void * blob)
{
	struct CHAIN * _cp;
	_cp = malloc (sizeof (struct CHAIN) );
	if (NULL == _cp) {
		perror ("malloc");
		exit (EXIT_FAILURE);
	}
	/* assign value as expected */
	bzero (_cp, sizeof (struct CHAIN) );
	_cp -> id = id;
	_cp -> label = label;
	_cp -> blob = blob;
	return _cp;
}

/* Initialize a new chain */
struct CHAIN *
chain_init (void)
{
	struct CHAIN * head;
	head = chain_create(0, "HEAD", NULL);
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
		printf ("E: chain_append(): invalid tailnew\n");
		exit (EXIT_FAILURE);
	}
	/* move to the last node */
	struct CHAIN * _cp;
	_cp = _chain_tail (head);
	/* append the tailnew after the last node */
	tailnew -> next = NULL;
	tailnew -> prev = _cp;
	_cp -> next = tailnew;
	/* chain_append returns * of tailnew */	
	return tailnew;
}

struct CHAIN *
chain_fastappend (struct CHAIN * head, char * label, void * blob)
{
	/* move to tail of chain */
	struct CHAIN * _cp, * _cursor;
	_cursor = _chain_tail (head);
	_cp = chain_create ( (_cursor -> id) + 1, label, blob );
	/* create and append a new chain */
	chain_append (head, _cp);
	return _cp;
}

void *
chain_destroy (struct CHAIN * head)
{
	/* check if the head is valid */
	if (NULL == head) {
		printf ("E: chain_destroy(): invalid head\n");
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
