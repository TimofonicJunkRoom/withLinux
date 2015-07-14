/* libstack
   2015 Lumin <cdluminate@gmail.com>
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

void
chain_print (struct CHAIN * node)
{
	/* check if node is valid */
	if (NULL == node) {
		printf ("* NODE (NULL) : NULL\n");
		return;
	}
	/* print the node */
	printf ("* NODE # (%d) @ %p\n"
			"       : prev  = %p , next = %p\n"
			"       : label = %s , blob = %p\n",
			node -> id, node,
			node -> prev, node -> next,
			node -> label, node -> blob);
	return;
}

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

struct CHAIN *
_chain_head (struct CHAIN * node)
{
	/* check if head is valid */
	if (NULL == node) {
		printf ("E: _chain_head(): NULL == node\n");
		exit (EXIT_FAILURE);
	}
	/* move to the last node */
	struct CHAIN * _cp;
	_cp = node;
	while (NULL != 	_cp -> prev) {
		_cp = _cp -> prev;
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

struct CHAIN *
chain_kill (struct CHAIN * node)
{
	struct CHAIN * cp;
	cp = NULL;
	/* check if node is NULL */
	if (NULL == node)
		return NULL;

	if (NULL != node -> prev) {
		cp = node -> prev;
		node -> prev -> next = NULL;
	}
	/* free node -> blob first, if not NULL */
	if (NULL != node -> blob)
		free (node -> blob);
	bzero (node, sizeof (struct CHAIN));
	free (node);
	return cp;
}

/* Initialize a new chain */
struct CHAIN *
chain_init (char * label, void * blob)
{
	struct CHAIN * head;
	head = chain_create(0, label, blob);
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

void
chain_dump (struct CHAIN * node)
{
	struct CHAIN * cp;
	cp = _chain_head (node);
	do {
		chain_print (cp);
		cp = cp -> next;
	} while (NULL != cp);
	return;
}

void *
chain_destroy (struct CHAIN * head)
{
	struct CHAIN * _cp;
	_cp = _chain_tail (head);
	/* free until NULL == prev */
	struct CHAIN * _prev;
	do {
		_prev = _cp -> prev;
		free (_cp);
		_cp = _prev;
	} while (NULL != _cp);
	return NULL;
}
/* vim : set ts = 4 */
