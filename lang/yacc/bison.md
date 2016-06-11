Bison Note
===

> file:///usr/share/doc/bison-doc/html/  
> bison user manual

# concepts

* _Language and grammar_  

`context-free grammar` description is a must in order for bison to parse a language.
`syntactic groupings` e.g. `expression`. Rules are often recursive. Any grammar described
in `Backus-Naur Form` or “BNF” is a context-free grammar. Bison reads machine-readable BNF.
Bison is optimized for LR grammars.

```C
/* a C function subdivided into tokens */

int               // keyword "int"

square (int x)    // identifier, open-paren, keyword "int", identifier, close-paren

{                 // open-brace

  return x * x;   // keyword "return", identifier, asterisk, identifier, semicolon

}                 // close-brace
```

* _grammar in bison_

nondeterminal symbol should be in lower case, while terminal symbol should be upper case.
e.g.
```
stmt: RETURN expr ';' ;
```

* _semantic values_

Each token in a bison grammar should have both a token type and a semantic value.

* _semantic actions_

```
expr: expr '+' expr    { $$ = $1 + $3; } ;
```

* GLR parsers

`reduce/reduce` conflicts and `shift/reduce` conflicts. Using `%glr-parser`
among bison declarations results in a Generalized LR parser.

Using GLR on unambigious grammar
```
// simple pascal type declarations
type subrange = lo .. hi;
type enum = (a, b, c);
type subrange = (a) .. b;
type enum = (a);

// bison parser
%token TYPE DOTDOT ID

%left '+' '-'
%left '*' '/'

%%
type_decl: TYPE ID '=' type ';' ;

type:
  '(' id_list ')'
| expr DOTDOT expr
;

id_list:
  ID
| id_list ',' ID
;

expr:
  '(' expr ')'
| expr '+' expr
| expr '-' expr
| expr '*' expr
| expr '/' expr
| ID
;
```

Using GLR to solve ambigious

* locations

* bison parser

```
The tokens come from a function called the lexical analyzer that you must supply in some fashion (such as by writing it in C). The Bison parser calls the lexical analyzer each time it wants a new token. It doesn’t know what is “inside” the tokens (though their semantic values may reflect this). Typically the lexical analyzer makes the tokens by parsing characters of text, but Bison does not depend on this.
```
CITE bison manual chapter 1.7

```
// user should define this
int yylex(void)
// and this
void yyerror (const char *)

// generated yyparse() stands at the top
int
main (void)
{ 
  return yyparse();
}
```

* stages

```
// dev
1. Write bison grammar file
2. Write a lexical analyzer to process input and pass tokens to the parser.
3. Write a controlling function that calls the Bison-produced parser.
4. Write error-reporting routines.

// run
1. Run Bison on the grammar to produce the parser.
2. Compile the code output by Bison, as well as any other source files.
3. Link the object files to produce the finished product.
```

* grammar layout

```
%{
Prologue
%}

Bison declarations

%%
Grammar rules
%%
Epilogue
```

# Examples

* [RPN](./ex/rpn)  

 In each action, the pseudo-variable $$ stands for the semantic value for the grouping that the rule is going to construct. Assigning a value to $$ is the main job of most actions. CITE bison manual 2.1.2

* [INC](./ex/inc)

* error recovery
```
line:
  '\n'
| exp '\n'   { printf ("\t%.10g\n", $1); }
| error '\n' { yyerrok;                  }
;
```

* location tracking

TODO

# Bison File

--TODO--
