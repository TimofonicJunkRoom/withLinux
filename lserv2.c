/* lserv2.c

   lumin's server 2, which implements sort of server/client model.

   This is a test code, the author is reading
   <<UNIX Network Programming : Socket API>>

   C.D.Luminate <cdluminate@163.com>
   2014/10/{03,04} */

/* lsock.h */
#include "lsock.h"

#define PORT 2333

#define BANNER "Sort_of_Server2 :: Please input your instructions.\n"
#define BYE_MSG "From Server : BYE\n\0\0"
#define HTML "<html>It works!</html>\n"
#define INST_NA "--- Instruction Not Supported ---\n"

#define AUTH_FAIL "--- Auth Failure ---\n"
#define AUTH_SUCC "--- Auth Success ---\n"
#define SECRET "++ This is the secret information for authed user\n"
#define SEC_FAIL "++ You are not permitted to see that SEC\n"

#define USER "user"
#define PASS "pass"

/* ---------- vars -------------*/
/* debug flag */
int debug = 1;

/* general socket related vars */
int listenfd;
int connfd;
struct sockaddr_in clie_addr;
struct sockaddr_in serv_addr;
socklen_t cli_len;
pid_t c_pid;

/* remote and server information */
	char raddr[128];		//remote addr 
	unsigned short rport = 0;	//remote port
	char saddr[128];		//server addr
	unsigned short sport = 0;	//server port

/* instruction buffer */
	char inst[1024];
/* feed back buffer */
	char feed[1024];
/* store parsed instruction */
	char argv0[1024];
	char argv1[1024];
/* wait time counter */
	long wcounter = 0;
/* readn : store return value of read() */
	int readn = 0;

/* more functions */
/* authentication */
	/* auth_state , >0 means login, <0 not login, 0 not defined */
	int auth_st = -1; 
	char auth_name[] = USER;
	char auth_pass[] = PASS;

/* -------------functions ----------*/
/* service things */
int do_serv (FILE *fp, int connfd);
/* parse instruction from client */
int inst_parse (const char *src, char *argv0, char *argv1);
/* when SIGINT */
void do_sigint (int sig);
/* flush socket related info */
int flush_sock_re (void)
{
	bzero (raddr, sizeof(raddr));
	bzero (saddr, sizeof(saddr));
	bzero (&serv_addr, sizeof(serv_addr));
	bzero (&clie_addr, sizeof(clie_addr));
	return 0;
}
/* flush buffer */
int flush_buf (void)
{
	bzero (inst, sizeof(inst));
	bzero (feed, sizeof(feed));
	bzero (argv0, sizeof(argv0));
	bzero (argv1, sizeof(argv1));
	return 0;
}
/* flush authentication info */
int flush_auth (void)
{
	bzero (auth_pass, sizeof(auth_pass));
	bzero (auth_name, sizeof(auth_name));
	return 0;
}

/* =========== main ================= */	
int
main (int argc, char **argv)
{
	/* flush socket related info */
	flush_sock_re ();
	flush_buf ();
	flush_auth ();

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

/* service matter */
int
do_serv (FILE *fp, int connfd)
{
	/* do serv loop */
	while (1) {
		/* if read instruction from client success ,
		   print instruction and write feed back */
		if ( (readn=read(connfd, inst, 1023)) > 0) {
			/* print conn info */
			fprintf (stdout, "- %s:%d -> %s: INST %s",
				 raddr, rport, "Server" , inst);
			/* print the feed back in buffer,
			   then write to connfd, default on */
			snprintf (feed, 1023, "RECV %s", inst);
			write (connfd, feed, strlen(feed));
		} else if (readn < 0) {
			perror ("read");
			exit (EXIT_FAILURE);
		} else {
			/* readn = 0, the normal case */
			usleep (100);
			wcounter += 100;
			if (wcounter >= 10*1000) {
				printf ("timeout\n");
				Close (connfd);
				exit (EXIT_FAILURE);
			}
			continue;
		}
		wcounter = 0;
		/* after basic read and write done, the instruction
		   should be parsed */

		/* parse instruction then */
		inst_parse (inst, argv0, argv1);
		/* compare parsed instruction with some commands */
		if (!strncmp(argv0, "GET", 3)) {
			write (connfd, HTML, sizeof(HTML));
		} else if (strncmp(argv0, "quit", 4)==0 ||
			   strncmp(argv0, "QUIT", 4)==0) {
			write (connfd, BYE_MSG, sizeof(BYE_MSG));
			Close (connfd);
			exit (EXIT_SUCCESS);
		} else if (!strncmp(argv0, "USER", 4)) {
			if (strlen(argv1)>0) strncpy (auth_name, argv1, 16);
		} else if (!strncmp(argv0, "PASS", 4)) {
			if (strlen(argv1)>0) strncpy (auth_pass, argv1, 16);
		} else if (!strncmp(argv0, "SEC", 3)) {
			if (auth_st > 0) {
				write (connfd, SECRET, sizeof(SECRET));
			} else if (auth_st <= 0) {
				write (connfd, SEC_FAIL, sizeof(SEC_FAIL));
				printf ("+ WARN: %s attempted to read SEC but failed\n",
					raddr);
			}
		} else if (!strncmp(argv0, "LOGOUT", 6)) {
			auth_st = -1;
			flush_auth ();
		} else {
			/* instruction not supported */
			write (connfd, INST_NA, sizeof(INST_NA));
			continue;
		}
		/* user authentication stuff */
		if (auth_st < 0) {
			if (strlen(auth_name)>0 && strlen(auth_pass)>0) {
				if (strncmp(auth_name, USER, sizeof(USER))==0 &&
				    strncmp(auth_pass, PASS, sizeof(PASS))==0) {
					auth_st = 1;
					write (connfd, AUTH_SUCC, sizeof(AUTH_SUCC));
				} else {
					write (connfd, AUTH_FAIL, sizeof(AUTH_FAIL));
					flush_auth ();
				}
			}
		}
		/* flush  buffers */
		flush_buf ();
	}
	return 0;
}

/* parse the instruction into argv0 and argv1 */
int
inst_parse (const char *src, char *argv0, char *argv1)
{
	sscanf (src, "%s %s", argv0, argv1);
	if (debug>1) printf ("+ argv0 [%s], argv1 [%s]\n", argv0, argv1);
	return 0;
}

/* when catched SIGINT */
void do_sigint (int sig)
{
	printf ("\n* Signal SIGINT\n");
	exit (EXIT_SUCCESS);
}
