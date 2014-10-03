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
char raddr[128];
unsigned short rport = 0;
char saddr[128];
unsigned short sport = 0;

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
	inet_ntop (AF_INET, &serv_addr.sin_addr, saddr, sizeof(serv_addr));
	sport = ntohs (serv_addr.sin_port);

	Bind (listenfd, (struct sockaddr *)&serv_addr,
	      sizeof(serv_addr));
	if (debug) printf ("* bind sucess\n");

	Listen (listenfd, 5);
	if (debug) printf ("* listenning on %s:%d ...\n",
		saddr , ntohs(serv_addr.sin_port));	

	while (1) {
		cli_len = sizeof(clie_addr);
		connfd = Accept(listenfd,
				(struct sockaddr *)&clie_addr,
				&cli_len);
		/* get client info */
		inet_ntop (AF_INET, &clie_addr.sin_addr.s_addr, raddr,
			sizeof(clie_addr));
		rport = ntohs(clie_addr.sin_port);

		/* fork a server */
		if ( (c_pid = Fork()) == 0) {
			Close (listenfd);
			write (connfd, BANNER,
			       sizeof(BANNER));
			do_serv (stdin, connfd); 
			write (connfd,
			       BYE_MSG, sizeof(BYE_MSG));
			Close (connfd);
			exit (0);
		}
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
		/* read instruction from client,
		   and feed back the instruction */
		if ( read(connfd, inst, 1023) > 0) {
			fprintf (stdout, "- %s:%d -> %s: INST %s",
				 raddr, rport,
				 "Server" , inst);
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

