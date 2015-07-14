struct CHAINPP {
	void           (* print             )(struct CHAIN * node);
	struct CHAIN * (* _tail             )(struct CHAIN * head);
	struct CHAIN * (* _head             )(struct CHAIN * node);
	struct CHAIN * (* create            )(int id, char * label, void * blob);
	struct CHAIN * (* init              )(char * label, void * blob);
	struct CHAIN * (* append            )(struct CHAIN * head, struct CHAIN * tailnew);
	struct CHAIN * (* fastappend        )(struct CHAIN * head, char * label, void * blob);
	void           (* dump              )(struct CHAIN * node);
	void *         (* destroy           )(struct CHAIN * head);
} chainpp ;

chainpp.print     = chain_print;
chainpp._tail     = _chain_tail;
chainpp._head     = _chain_head;
chainpp.create    = chain_create;
chainpp.init      = chain_init;
chainpp.append    = chain_append;
chainpp.fastappend= chain_fastappend;
chainpp.dump      = chain_dump;
chainpp.destroy   = chain_destroy;

