#include "lsock.h"

#define PORT 2333
#define BANNER "hello!socket\n"

int
main (int argc, char **argv)
{
	int	listenfd, connfd;
	pid_t	c_pid;
	socklen_t	cli_len;
	struct sockaddr_in clie_addr, serv_addr;

	/* create socket */
	listenfd = Socket(AF_INET,
			  SOCK_STREAM, 0);

	/* fill in sockaddr */
	bzero (&serv_addr, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	serv_addr.sin_port = htons(PORT);

	Bind (listenfd, (struct sockaddr *)&serv_addr,
	      sizeof(serv_addr));

	Listen (listenfd, 5);

	while (1) {
		cli_len = sizeof(clie_addr);
		connfd = Accept(listenfd,
				(struct sockaddr *)&clie_addr,
				&cli_len);
		if ( (c_pid = Fork()) == 0) {
			Close (listenfd);
			write (connfd,
			       BANNER,
			       sizeof(BANNER));
			exit (0);
		}
		Close (connfd);
	}

	return 0;
}
