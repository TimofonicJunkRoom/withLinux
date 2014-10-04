#include "lsock.h"

#define PORT 2333
#define BANNER "hello!socket\n"
#define BYE_MSG "From Server : BYE\n"

/* debug flag */
int debug = 1;

/* general vars */
int listenfd, connfd;
pid_t c_pid;
socklen_t cli_len;

struct sockaddr_in clie_addr;
struct sockaddr_in serv_addr;

/* remote and server information */
char raddr[128];		//remote addr 
unsigned short rport = 0;	//remote port
char saddr[128];		//server addr
unsigned short sport = 0;	//server port

/* service things */
int do_serv (FILE *fp, int connfd);

int
main (int argc, char **argv)
{
	/* flush buffers */
	bzero (raddr, 128);
	bzero (saddr, 128);
	bzero (&serv_addr, sizeof(serv_addr));
	bzero (&clie_addr, sizeof(clie_addr));

	/* create socket */
	listenfd = Socket(AF_INET,
			  SOCK_STREAM, 0);
	if (debug) printf ("* initialized socket\n");

	/* fill in sockaddr */
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	serv_addr.sin_port = htons(PORT);
	/* fill server ipv4 addr and port info */
	inet_ntop (AF_INET, &serv_addr.sin_addr, saddr, sizeof(serv_addr));
	sport = ntohs (serv_addr.sin_port);

	/* bind */
	Bind (listenfd, (struct sockaddr *)&serv_addr,
	      sizeof(serv_addr));
	if (debug) printf ("* bind sucess\n");
	
	/* listen */
	Listen (listenfd, 5);
	if (debug) printf ("* listenning on %s:%d ...\n",
		saddr , ntohs(serv_addr.sin_port));	

	/* standalone : wait and accept clients */
	while (1) {
		cli_len = sizeof(clie_addr);
		/* if no client connects the server, 
		   the function will block server */
		connfd = Accept(listenfd,
				(struct sockaddr *)&clie_addr,
				&cli_len);
		/* get client addr and port info */
		inet_ntop (AF_INET, &clie_addr.sin_addr.s_addr, raddr,
			sizeof(clie_addr));
		rport = ntohs(clie_addr.sin_port);

		/* fork the server client,
		   the parent process closes connfd and listen to other
		   clients, the child process do service matter and exit*/
		if ( (c_pid = Fork()) == 0) {
			/* c_pid in parent process will not be 0,
			   so it will not step in here.
			   c_pid in child process will be 0 or -1(error),
			   so code here tells child what to do */
			/* listen mode off */
			Close (listenfd);
			/* write welcome message back */
			write (connfd, BANNER,
			       sizeof(BANNER));
			/* start do_serv loop, until instruction
			   'quit' is recieved */
			do_serv (stdin, connfd); 
			/* had recieved the 'quit' instruction, do_serv
			   loop was broke. then write the BYE_BYE message */
			write (connfd,
			       BYE_MSG, sizeof(BYE_MSG));

			/* close connfd and exit */
			Close (connfd);
			exit (EXIT_SUCCESS);
		}
		/* parent process will not step into the if () code block,
		   so parent should just close connfd, and let the child
		   do_serv */
		Close (connfd);
	}

	return 0;
}

int
do_serv (FILE *fp, int connfd)
{
	/* instruction buffer */
	char inst[1024];
	bzero (inst, 1024);
	/* feed back buffer */
	char feed[1024];
	bzero (feed, 1024);

	/* do serv loop */
	while (1) {
		/* if read instruction from client success ,
		   print instruction and write feed back */
		if ( read(connfd, inst, 1023) > 0) {
			/* print conn info */
			fprintf (stdout, "- %s:%d -> %s: INST %s",
				 raddr, rport,
				 "Server" , inst);
			/* print the feed back in buffer,
			   then write to connfd */
			snprintf (feed, 1023, "- %s:%d : RECV %s",
				  saddr, sport, inst);
			write (connfd, feed, strlen(feed));
		}
		/* if encountered the quit instruction */
		if (strncmp(inst, "quit", 4) == 0 ||
		    strncmp(inst, "QUIT", 4) == 0) {
			break;
		}
		/* flush instructions */
		bzero (inst, 1024);
	}
	return 0;
}

