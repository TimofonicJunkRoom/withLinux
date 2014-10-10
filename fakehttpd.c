/* fakehttpd.c
   Linux / GCC

   Nearly a extremely tiny httpd, but
   in fact it is a fake.

   C.D.Luminate

   2014/10/08
   Licence : MIT
   */
/*
   TODO add signal handle
   TODO add error handle
   TODO add debug printf's
   TODO add log module
   TODO enhance security
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

/* USAGE / HELP */
int Usage (const char *argv0) {
	fprintf (stderr, "\
Fakehttpd, version devel\n\
Author: C.D.Luminate / MIT Licence / 2014\n\
Usage:  %s [options] FILE\n\
options:\n\
	-h	show this help message\n\
	-p PORT	specify a port number\n\
	*-v	verbose/debug output\n\
	*-b	set the addr to bind\n\
	...\n\
To be continued\n", argv0);
	return 0;
}

/* FLAGS */
/* TODO someday, set debug to 0 as default */
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
int serverport = PORT; /* default 8000 */
int fileind; /* file name indicator, = optind */

pid_t child_pid;

int opt;

/* FUNCTIONS */
int httpd_serve (const char *pathname, int connfd); /* all httpd matter */
int httpd_parse_req (const char *req, char *filename); /* parse request */

/* *MAIN* */
int
main (int argc, char **argv)
{
	/* parse argv with getopt() */
	while ( (opt = getopt(argc, argv, "hp:v")) != -1) {
		switch (opt) {
			case 'h':
				Usage (argv[0]);
				exit (EXIT_SUCCESS);
				break;
			case 'p':
				serverport = atoi(optarg);
				/* see if the given number is valid */
				if (serverport < 1 ||
				    serverport > 65535) {
					fputs ("Invalid Port Number.\n", stderr);
					exit (EXIT_FAILURE);
				}
				break;
			case 'v':
				debug = 1;
				break;
			default:
				Usage (argv[0]);
				exit (EXIT_FAILURE);
		}
	}
				
	/* only a open() test : open specified file,
	   note that the filename is the last argument */
	fileind = optind;
	if (fileind >= argc) {
		fprintf (stderr, "%s: Expected Filename\n", argv[0]);
		exit (EXIT_FAILURE);
	}
	if ((openfd=open(argv[fileind], O_RDONLY)) == -1) {
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
	serveraddr.sin_port = htons((short)serverport);
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
			httpd_serve (argv[fileind], connfd);
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
	/* prepare buffer and var */
	char request[1024];
	char req_fname[1024];

	char response[1024];

	bzero (request, 1024);
	bzero (req_fname, 1024);

	/* read the request, then parse it with
	   httpd_parse_req, which returns the HTTP
	   status code */
	if ( read(connfd, request, 1023) == -1) {
		perror ("read");
		exit (EXIT_FAILURE);
	}
	switch (httpd_parse_req (request, req_fname)) {
		case 200:
			/* HTTP/1.0 200 OK */
			break;
		default:
			/* UNKNOWN */
			close (connfd);
			exit (EXIT_FAILURE);
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

int
httpd_parse_req (const char *request, char *req_fname)
{
	if (strncmp(request, "GET", 3) != 0) {
#define BADREQ "500 Bad Request"
		write (connfd, BADREQ, sizeof(BADREQ));
		close (connfd);
		exit (EXIT_SUCCESS);
	}
	return 200;
}

