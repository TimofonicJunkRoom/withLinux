#include "lsock.h"

#define RPORT 2333

/* buffer, read from server */
char buffer[1024];

int do_clie (FILE *, int);
void do_sigint (int sig);


int	sockfd;

/* main */
int
main (int argc, char **argv)
{
	struct sockaddr_in serv_addr;

	bzero (buffer, 1024);

	/* test arg */
	if (argc != 2) {
		exit (EXIT_FAILURE);
	}

	/* create socket */
	sockfd = Socket (AF_INET, SOCK_STREAM, 0);

	/* fill sockaddr_in */
	bzero (&serv_addr, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(RPORT);
	inet_pton(AF_INET, argv[1], &serv_addr.sin_addr);

	/* connect */
	Connect (sockfd, (struct sockaddr *)&serv_addr,
		 sizeof(serv_addr));

	//read (sockfd, buffer, 1024);
	//printf ("from server : %s\n", buffer);
	//bzero (buffer, 1024);
	//read (sockfd, buffer, 2014);
	//printf ("from server : %s\n", buffer);

	/*long n = 0;
	while ( (n = read(sockfd, buffer, 1024-1)) != 0) {
		printf ("from server : %s\n", buffer);
		bzero (buffer, 1024);
	}*/
	(void) signal (SIGINT, do_sigint);
	do_clie (stdin, sockfd);

	return 0;
}

int
do_clie (FILE *fp, int sockfd)
{
	char sendline[1024];
	bzero (sendline, 1024);

	int readn = 0;
	while (1) {
		readn = read(sockfd, buffer, 1023);
		printf ("%s", buffer);
		bzero (buffer, 1024);
		//readn = read(sockfd, buffer, 1023);
		//printf ("%s", buffer);
		//bzero (buffer, 1024);
			

		fgets (sendline, 1023, stdin);
		write (sockfd, sendline, strlen(sendline));
		bzero (sendline, 1024);

		if (!strncmp(sendline, "quit", 4)) {
			Close (sockfd);
			printf ("quit\n");
			exit (EXIT_SUCCESS);
		}
	}
	Close (sockfd);
	return 0;
}

void do_sigint (int sig)
{
	printf ("quit\n");
	write (sockfd, "QUIT", 5);
	Close (sockfd);
	exit (EXIT_SUCCESS);
}
