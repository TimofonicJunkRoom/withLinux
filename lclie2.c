#include "lsock.h"

#define RPORT 2333

int str_clie (FILE *, int);

int
main (int argc, char **argv)
{
	int	sockfd;
	struct sockaddr_in serv_addr;

	char buffer[1024];
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

	read (sockfd, buffer, 1024);
	printf ("from server : %s\n", buffer);
	//bzero (buffer, 1024);
	//read (sockfd, buffer, 2014);
	//printf ("from server : %s\n", buffer);

	/*long n = 0;
	while ( (n = read(sockfd, buffer, 1024-1)) != 0) {
		printf ("from server : %s\n", buffer);
		bzero (buffer, 1024);
	}*/
	str_clie (stdin, sockfd);

	return 0;
}

int
str_clie (FILE *fp, int sockfd)
{
	char sendline[1024], recvline[1024];
	bzero (sendline, 1024);
	bzero (recvline, 1024);

	while (fgets(sendline, 1023, fp) != NULL) {
		write (sockfd, sendline, strlen(sendline));
		fputs (recvline, stdout);
	}
	return 0;
}

