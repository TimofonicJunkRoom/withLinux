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

	printf ("chain_dump ()\n");
	chain_dump (cp);

	chain_fastappend (cp, "Gauss", NULL);
	chain_fastappend (cp, "Fermi", NULL);
	chain_fastappend (cp, "Galois", NULL);
	chain_fastappend (root, "Fourier", NULL);
	chain_dump (cp);

	chain_destroy (root);
	return 0;
}
	
