#include "chain.c"
#include "chainpp.c"

int
main (void)
{
	/* init chainpp extension */
	chainpp_init (&chainpp);

	struct CHAIN * root = NULL;
	struct CHAIN * cp   = NULL;

	/* print NULL */
	chain_print (cp);

	/* create chain */
	root = chain_init ("HEAD", NULL);
	chain_print (root);
	printf ("### root node created \n");
	
	/* adding tail */
	cp = chain_fastappend (root, "Laplace", NULL);
	chain_fastappend (cp, "Gauss", NULL);
	chain_fastappend (cp, "Fermi", NULL);
	chainpp.fastappend (root, "Lumin", NULL);
	chain_fastappend (cp, "Galois", NULL);
	cp = chain_fastappend (root, "Fourier", NULL);

	/* dump */
	printf ("############ chainpp.dump ()\n");
	chainpp.dump (root);

	/* remove last */
	cp = chain_kill (cp);

	/* dump again */
	chain_genindex (root);
	chainpp.dump (root);

	chainpp.print (chain_pick (root, 2));

	/* destroy */
	chain_destroy (root);
	return 0;
}
	
