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
 mount.ecryptfs <SRC> <DST> <STUB> <STUB> -- Mount SRC to DST, Enhanced\n\
 mount.ecryptfs <SRC> <DST> <STUB>        -- Mount SRC to DST by Shortcut\n\
 mount.ecryptfs <SRC> <DST>               -- Mount SRC to DST\n\
 mount.ecryptfs <DST>                     -- Umount DST\n");
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
  } else if (argc == 4) { // Shortcut mount
    printf ("mount -t ecryptfs -o *** %s %s\n", argv[1], argv[2]);
    setuid(geteuid());
    char * shortcut_opt="-o key=passphrase:,ecryptfs_cipher=aes,ecryptfs_key_bytes=16,ecryptfs_passthrough=n,ecryptfs_enable_filename_crypto=n,no_sig_cache";
    char * mount_ecryptfs_argv[] = {
        "mount", "-t", "ecryptfs", argv[1], argv[2], "-o", shortcut_opt, NULL
    };
    execve ("/bin/mount", mount_ecryptfs_argv, envp);
  } else if (argc == 5) { // Enhanced shortcut mount
    printf ("mount -t ecryptfs -o *** %s %s\n", argv[1], argv[2]);
    setuid(geteuid());
    char * shortcut_opt = ""
"key=passphrase,ecryptfs_cipher=aes,ecryptfs_key_bytes=16,"
"ecryptfs_passthrough=n,ecryptfs_enable_filename_crypto=y,no_sig_cache,"
"ecryptfs_fnek_sig=d395309aaad4de06"; // f("test") = d395309aaad4de06
    char * mount_ecryptfs_argv[] = {
        "mount", "-t", "ecryptfs", argv[1], argv[2], "-o", shortcut_opt, NULL
    };
    execve ("/bin/mount", mount_ecryptfs_argv, envp);
  } else {
    Usage ();
    return 1;
  }
  return 0;
}
