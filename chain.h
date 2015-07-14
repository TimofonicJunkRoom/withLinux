void
chain_print (struct CHAIN * node);


struct CHAIN *
_chain_tail (struct CHAIN * head);


struct CHAIN *
chain_create (int id, char * label, void * blob);


struct CHAIN *
chain_init (char * label, void * blob);


struct CHAIN *
chain_append (struct CHAIN * head, struct CHAIN * tailnew);


struct CHAIN *
chain_fastappend (struct CHAIN * head, char * label, void * blob);


void *
chain_destroy (struct CHAIN * head);

