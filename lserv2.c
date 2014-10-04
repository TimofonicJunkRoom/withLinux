#include "lsock.h"

#define PORT 2333
#define BANNER "Sort_of_Server2 :: Please input your instructions.\n"
#define BYE_MSG "From Server : BYE\n"
#define HTML "<html>It works!</html>\n"

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
/* parse instruction from client */
int inst_parse (const char *src, char *argv0, char *argv1);
/* */
void do_sigint (int sig);

/* --- main --- */	
int
main (int argc, char **argv)
{
	/* flush buffers */
	bzero (raddr, 128);
	bzero (saddr, 128);
	bzero (&serv_addr, sizeof(serv_addr));
	bzero (&clie_addr, sizeof(clie_addr));

	/* create socket */
	listenfd = Socket(AF_INET, SOCK_STREAM, 0);
	if (debug) printf ("* initialized socket\n");

	/* fill in sockaddr */
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	serv_addr.sin_port = htons(PORT);
	/* fill server ipv4 addr and port info */
	inet_ntop (AF_INET, &serv_addr.sin_addr, saddr, sizeof(serv_addr));
	sport = ntohs (serv_addr.sin_port);

	/* bind */
	Bind (listenfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
	if (debug) printf ("* bind sucess\n");
	
	/* listen */
	Listen (listenfd, 5);
	if (debug) printf ("* listenning on %s:%d ...\n",
		saddr , ntohs(serv_addr.sin_port));	

	/* standalone : wait and accept clients */
	(void) signal(SIGINT, do_sigint);
	while (1) {
		cli_len = sizeof(clie_addr);
		/* if no client connects the server, 
		   the function will block server */
		connfd = Accept(listenfd,
				(struct sockaddr *)&clie_addr, &cli_len);
		/* get client addr and port info */
		inet_ntop (AF_INET, &clie_addr.sin_addr.s_addr, raddr,
			sizeof(clie_addr));
		rport = ntohs(clie_addr.sin_port);

		if (debug) printf ("* accept client from %s:%d\n",
				   raddr, rport);

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
			write (connfd, BANNER, sizeof(BANNER));
			/* start do_serv loop, until instruction
			   'quit' is recieved */
			do_serv (stdin, connfd); 
			/* had recieved the 'quit' instruction, do_serv
			   loop was broke. then write the BYE_BYE message */
			write (connfd, BYE_MSG, sizeof(BYE_MSG));

			/* close connfd and exit */
			Close (connfd);
			exit (EXIT_SUCCESS);
		}
		/* parent process will not step into the if () code block,
		   so parent should just close connfd, and let the child
		   do_serv */
		Close (connfd);
	}

	Close (listenfd);
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
	/* store parsed instruction */
	char argv0[1024];
	char argv1[1024];
	bzero (argv0, 1024);
	bzero (argv1, 1024);
	/* wait time counter */
	long wcounter = 0;

	/* readn : store return value of read() */
	int readn = 0;

	/* more functions */
#define USER "user"
#define PASS "password"
	int auth_st = -1; //auth_state not authenticated.
	char auth_name[] = USER;
	bzero (auth_name, sizeof(auth_name));
	char auth_pass[] = PASS;
	bzero (auth_pass, sizeof(auth_pass));

	/* do serv loop */
	while (1) {
		/* if read instruction from client success ,
		   print instruction and write feed back */
		if ( (readn=read(connfd, inst, 1023)) > 0) {
			/* print conn info */
			fprintf (stdout, "- %s:%d -> %s: INST %s",
				 raddr, rport, "Server" , inst);
			/* print the feed back in buffer,
			   then write to connfd */
			snprintf (feed, 1023, "- %s:%d : RECV %s",
				  saddr, sport, inst);
			write (connfd, feed, strlen(feed));
			wcounter = 0;
		} else if (readn < 0) {
			perror ("read");
			exit (EXIT_FAILURE);
		} else {
			/* readn = 0, the normal case */
			usleep (100);
			wcounter += 100;
			continue;
		}
		/* after basic read and write done, the instruction
		   should be parsed */

		/* if encountered the quit instruction */
		if (strncmp(inst, "quit", 4) == 0 ||
		    strncmp(inst, "QUIT", 4) == 0) {
			/* had recieved the 'quit' instruction, do_serv
			   loop was broke. then write the BYE_BYE message */
			write (connfd, BYE_MSG, sizeof(BYE_MSG));

			/* close connfd and exit */
			Close (connfd);
			exit (EXIT_SUCCESS);
		}
		/* other instructions */
		inst_parse (inst, argv0, argv1);
		if (!strncmp(argv0, "GET", 3)) {
			write (connfd, HTML, sizeof(HTML));
		} else if (!strncmp(argv0, "USER", 4)) {
			strncpy (auth_name, argv1, 16);
		} else if (!strncmp(argv0, "PASS", 4)) {
			strncpy (auth_pass, argv1, 16);
		} else if (!strncmp(argv0, "SEC", 3)) {
#define SECRET "++ This is the secret information for authed user\n"
#define SEC_FAIL "++ You are not permitted to see that SEC\n"
			if (auth_st > 0) {
				write (connfd, SECRET, sizeof(SECRET));
			} else if (auth_st <= 0) {
				write (connfd, SEC_FAIL, sizeof(SEC_FAIL));
				printf ("+ %s attempted to read SEC but failed\n",
					raddr);
			}
		} else if (!strncmp(argv0, "LOGOUT", 6)) {
			auth_st = -1;
			bzero (auth_name, sizeof(auth_name));
			bzero (auth_pass, sizeof(auth_pass));
		}

		/* user authentication stuff */
		if (auth_st < 0) {
#define AUTH_SUCC "- Auth Success\n"
#define AUTH_FAIL "- Auth Failure\n"
			if (strlen(auth_name)>0 && strlen(auth_pass)>0) {
				if (strncmp(auth_name, USER, sizeof(USER))==0 &&
				    strncmp(auth_pass, PASS, sizeof(PASS))==0) {
					auth_st = 1;
					write (connfd, AUTH_SUCC, sizeof(AUTH_SUCC));
				} else {
					write (connfd, AUTH_FAIL, sizeof(AUTH_FAIL));
					bzero (auth_name, sizeof(auth_name));
					bzero (auth_pass, sizeof(auth_pass));
				}
			}
		}


		/* flush instructions and other buffers */
		bzero (inst, 1024);
		bzero (feed, 1024);
		bzero (argv0, 1024);
		bzero (argv1, 1024);
	}
	return 0;
}

/* parse some instruction */
int
inst_parse (const char *src, char *argv0, char *argv1)
{
	sscanf (src, "%s %s", argv0, argv1);
	if (debug) printf ("+ argv0 [%s], argv1 [%s]\n", argv0, argv1);
	return 0;
}

/* when catched SIGINT */
void do_sigint (int sig)
{
	printf ("\n* Catch SIGINT, exit.\n");
	exit (EXIT_SUCCESS);
}

