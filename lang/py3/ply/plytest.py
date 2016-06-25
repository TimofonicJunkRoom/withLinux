#!/usr/bin/python3
'''
python3-ply test
@ reference: package python-ply-doc
@ reference: http://www.dabeaz.com/ply/ply.html
'''
import ply.lex as lex
import ply.yacc as yacc

'''
lex part
'''
def BuildLexer():
  print('initialize lexer')
  tokens = (
    'NUMBER',
    'PLUS',
    'MINUS',
    'TIMES',
    'DIVIDE',
    'LPAREN',
    'RPAREN',
  )

  t_PLUS    = r'\+'
  t_MINUS   = r'-'
  t_TIMES   = r'\*'
  t_DIVIDE  = r'/'
  t_LPAREN  = r'\('
  t_RPAREN  = r'\)'

  def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)    
    return t
  
  def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)
  
  t_ignore  = ' \t'
  
  def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)
  
  lexer = lex.lex()
  lexer.tokens = tokens
  return lexer

lexer = BuildLexer()

print('unit test: lexer')
def unit_lexer():
  data = '''
3 + 4 * 10
  + -20 *2
'''
  lexer.input(data)
  for tok in lexer:
    print(tok)
    print(tok.type, tok.value, tok.lineno, tok.lexpos)
unit_lexer()
