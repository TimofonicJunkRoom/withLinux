/* fakehttpd.c

   Nearly a extremely tiny httpd, but
   in fact it is a fake.

   C.D.Luminate

   2014/10/08
   Licence : MIT
   */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#define PORT 8000

/* FLAGS */
int debug = 1;

/* VARIABLES */
int openfd; /* file descriptor returned by open() */
char buffer[1024]; /* buffer of read() */
int readn; /* return value of read() */
long content_length; /* size of file opened */

struct sockaddr_in serveraddr;
struct sockaddr_in clientaddr;
int sockfd; /* socket file descriptor to listen */
int connfd; /* connection file descriptor */
socklen_t client_len;

pid_t child_pid;

/* FUNCTIONS */
int httpd_serve (const char *pathname, int connfd);

/* *MAIN* */
int
main (int argc, char **argv)
{
	/* parse argv */
	if (argc != 2) {
		exit (-1);
	}

	/* only a open() test : open specified file */
	if ((openfd=open(argv[1], O_RDONLY)) == -1) {
		perror ("open");
		exit (EXIT_FAILURE);
	}
	/* count content_length */
	content_length = 0;
	while ( (readn = read(openfd, buffer, 1024)) > 0) {
		content_length += readn;
	}
	bzero (buffer, 1024);
	readn = 0;
	close (openfd);
	if (debug) printf ("Content-Length: %ld\n", content_length);

	/* create socket */
	sockfd = socket(AF_INET, SOCK_STREAM, 6);
	if ( sockfd == -1) {
		perror ("socket");
		exit (EXIT_FAILURE);
	}
	/* fill sockaddr structure */
	bzero (&serveraddr, sizeof(serveraddr));
	serveraddr.sin_family = AF_INET;
	serveraddr.sin_port = htons(PORT);
	inet_pton (AF_INET, "127.0.0.1", &serveraddr.sin_addr);
	/* bind */
	if (bind(sockfd, (struct sockaddr *)&serveraddr,
	    sizeof(serveraddr)) == -1) {
		perror ("bind");
		exit (EXIT_FAILURE);
	}
	/* listen */
	if (listen(sockfd, 8) == -1) {
		perror ("listen");
		exit (EXIT_FAILURE);
	}

	/* wait and accept */
	while (1) {
		client_len = sizeof(clientaddr);
		/* server blocked at accept() */
		connfd = accept (sockfd, (struct sockaddr *)&clientaddr,
				 &client_len);
		if (connfd == -1) {
			perror ("accept");
			exit (EXIT_FAILURE);
		}
		/* when accepted client, fork a server */
		if ( (child_pid = fork()) == 0) {
			/* only child do this */
			close (sockfd);
			/* do fakehttpd major matter and exit */
			httpd_serve (argv[1], connfd);
			exit (EXIT_SUCCESS);
		}
		/* the parent process will step here, instead of if(){}
		   block above .for parent server, close connfd and 
		   wait for next connection */
		close (connfd);
	}

	/* close file and return */
	close (openfd);
	close (sockfd);
	return 0;
}

int
httpd_serve (const char *pathname, int connfd)
{
	char request[1024];
	char response[1024];

	/* parse request */
	bzero (request, 1024);
	if ( read(connfd, request, 1023) == -1) {
		perror ("read");
		exit (EXIT_FAILURE);
	}
	if (strncmp(request, "GET", 3) != 0) {
#define BADREQ "500 Bad Request"
		write (connfd, BADREQ, sizeof(BADREQ));
		close (connfd);
		exit (EXIT_SUCCESS);
	}

	/* send response header */
#define RESPONSE "HTTP/1.0 200 OK\n\
Date:\n\
Server: Fakehttpd\n\
Content-Length: %ld\n\
Content-Type: text/html; charset=utf-8\n\
Connection: keep-alive\n\
\n\n"
	snprintf (response, 1024, RESPONSE, content_length);
	write (connfd, response, strnlen(response, 1024));

	/* open the file specified, the 2nd time, then dump into connfd */
	openfd = open (pathname, O_RDONLY);
	bzero (buffer, 1024);
	while ( (readn = read(openfd, buffer, 1024)) > 0) {
		write (connfd, buffer, strnlen(buffer, 1024));
		bzero (buffer, 1024);
	}

	/* close */
	close (openfd);
	close (connfd);
	//shutdown (connfd, SHUT_RDWR);
	return 0;
}
