/* 
 * example
 */

%{
  #include <stdio.h>

  char *
  rmnewline (char * head) {
    char * s = head;
    while (*(s++) != '\0') {
      if (*s == '\n') {
        //printf ("[%c]", *s);
        *s = '\0';
      }
    }
    return head;
  }

  void
  yyerror(char * message) {
    printf("\nError: %s\n",message);
  }

%}

spaces [ \t]
number [0-9.]

%option noyywrap

%%

{number}+{spaces}*\n {
  printf ("%s;\n", rmnewline(yytext)); }
^{number}+ {
  printf ("  %s, ", yytext); }
{number}+ {
  printf ("%s, ", yytext); }
[:alpha:] {
  yyerror("Illegal Character"); }
<<EOF>> {
  printf ("]\n", yytext); exit(0); }

%%

int
main (int argc, char **argv)
{
  printf ("data = [\n");
  yylex();
  return 0;
}
