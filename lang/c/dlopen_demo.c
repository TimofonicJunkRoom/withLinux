/* dlopen demo
  @ref dlopen(3)
  @compile cc *.c -ldl
 */

#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <gnu/lib-names.h> // for LIBM_SO

int
main(void)
{
  void * handle;
  double (*cosine)(double);
  char * error;

  handle = dlopen(LIBM_SO, RTLD_LAZY);
  if (!handle) {
    fprintf(stderr, "%s\n", dlerror());
    exit(EXIT_FAILURE);
  }
  dlerror();

  cosine = (double (*)(double)) dlsym(handle, "cos");
  
  error = dlerror();
  if (error != NULL) {
    fprintf(stderr, "%s\n", error);
    exit(EXIT_FAILURE);
  }

  printf("%f\n", (*cosine)(2.0));
  dlclose(handle);
  return 0;
}
