/*

 A Sloppy Setuid Ecryptfs Wrapper

 Copyright (C) 2016 Zhou Mo <cdluminate@gmail.com>
 License: MIT License (Expat)

 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/*

Example:

$ gcc -Wall mount.ecryptfs.c -o mount.ecryptfs
$ ./mount.ecryptfs EC ec                        # fails
$ sudo chmod u+s mount.ecryptfs
$ sudo chown root:root mount.ecryptfs
$ ./mount.ecryptfs EC ec                        # OK
$ ./mount.ecryptfs ec                           # umount

 */

void
Usage ()
{
  printf("Usage:\n\
 mount.ecryptfs <SRC> <DST> -- Mount SRC to DST\n\
 mount.ecryptfs <DST>       -- Umount DST\n");
  return;
}

int
main (int argc, char ** argv, char ** envp)
{
  if (argc == 3) { // Mount
    printf ("mount -t ecryptfs %s %s\n", argv[1], argv[2]);
    setuid(geteuid());
    char * mount_ecryptfs_argv[] = {
        "mount", "-t", "ecryptfs", argv[1], argv[2], NULL
    };
    execve ("/bin/mount", mount_ecryptfs_argv, envp);
  } else if (argc == 2) { // Umount
    printf ("umount %s\n", argv[1]);
    setuid(geteuid());
    char * umount_ecryptfs_argv[] = { "umount", argv[1], NULL };
    execve ("/bin/umount", umount_ecryptfs_argv, envp);
  } else {
    Usage ();
    return 1;
  }
  return 0;
}
