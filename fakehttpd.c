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

#include <signal.h>

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

int opt; /* for getopt() */

/* FUNCTIONS */
int httpd_serve (const char *pathname, int connfd); /* all httpd matter */
int httpd_parse_req (const char *req, char *filename); /* parse request */
int httpd_resp_head (int connfd, int status); /* send response head */

/* Signal Handle */
struct sigaction act;
struct sigaction oldact;
void handle_sig (int sig);

/* *MAIN* */
int
main (int argc, char **argv)
{
	/* prepare signal actions */
	act.sa_handler = handle_sig;
	act.sa_flags = SA_RESETHAND | SA_NODEFER;
	sigaction (SIGINT, &act, &oldact);
	printf ("sig set \n");

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

	int http_status;

	bzero (request, 1024);
	bzero (req_fname, 1024);
	http_status = 0;

	/* read the request, then parse it with
	   httpd_parse_req, which returns the HTTP
	   status code, and it will put the requested filename
	   in the req_fname string */
	if ( read(connfd, request, 1023) == -1) {
		perror ("read");
		exit (EXIT_FAILURE);
	}
	http_status = httpd_parse_req (request, req_fname);

	/* send response header */
	httpd_resp_head (connfd, http_status);

	/* decide what to do after sending the header */
	switch (http_status) {
		case 200:
			/* HTTP/1.0 200 OK */
			break;
		case 501:
		default:
			/* UNKNOWN */
			close (connfd);
			exit (EXIT_SUCCESS);
	}
			

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
	/* strncmp the request */
	if (strncmp(request, "GET", 3)==0) {
	//if (strncmp(request, "GET / HTTP/1.0", 14)==0) {
		/* do the default */
		return 200;
	} else {
		return 501;
	}
	return -1;
}

int
httpd_resp_head (int connfd, int _hstatus)
{
	/* prepare response head */
	char response[1024];
	bzero (response, 1024);

/* General Header */
#define H_A \
"HTTP/1.1 %s\n\
Date: %s\n\
Server: Fakehttpd\n\
Content-Length: %ld\n\
Connection: close\n\
Content-Type: %s\n\n"
/* 200 header */
#define H_200 "HTTP/1.1 200 OK\n\
Date:\n\
Server: Fakehttpd\n\
Content-Length: %ld\n\
Connection: close\n\
Content-Type: text/html; charset=utf-8\n\n"
/* 501 header */
#define H_501 "HTTP/1.1 501 Not Implemented\n\
Date:\n\
Server: Fakehttpd\n\
Content-Length: 0\n\
Connection: close\n\
Content-Type: text/html; charset=utf-8\n\n"

	/* send header according to http_status */
	switch (_hstatus) {
		case 200:
			/* 200 OK */
			snprintf (response, 1024, H_200, content_length);
			write (connfd, response, strnlen(response, 1024));
			break;
		case 501:
			/* 501 not implemented */
			snprintf (response, 1024, H_501);
			write (connfd, response, strnlen(response, 1024));
			break;
		default:
			return -1;
	}
	return 0;
}

void
handle_sig (int sig)
{
	printf ("signal SIGINT");
}
