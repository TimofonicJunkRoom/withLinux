#include "chain.c"

int
main (void)
{
	struct CHAIN * root = NULL;
	struct CHAIN * cp   = NULL;
	root = chain_init ("HEAD", NULL);
	chain_print (root);
	chain_print (cp);
	cp = chain_fastappend (root, "Laplace", NULL);
	chain_print (root);
	chain_print (cp);
	return 0;
}
	
