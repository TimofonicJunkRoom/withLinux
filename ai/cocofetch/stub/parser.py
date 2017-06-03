#!/usr/bin/python3
'''
Corenlp Text tree parser
Copyright (C) 2016 Zhou Mo

Parser in the lex + yacc manner.
'''
import sys
import lumin_log as log

debug = 1
log.info('Tree parser and converter, debug = %d' % debug)

yaccspec='''
tnode:
  (ATTR WORD)
| (ATTR tnode*)
;
'''

if debug:
  log.info('dump language spec')
  print(yaccspec)

if debug: log.debug('initialize Tnode')
class Tnode(object):
  def __init__(self, attr, content=None):
    self.attr_ = attr
    self.content_ = content
    #self.content = word or unit list
  def dump(self, indent=0):
    assert(self.content_)
    if isinstance(self.content_, str):
      print('%s(%s %s)'%(' '*indent, self.attr_, self.content_))
    else:
      print('%s(%s'%(' '*indent, self.attr_))
      for item in self.content_:
        item.dump(indent+2)
      print('%s)'%(' '*indent))
  def setcontent(self, content):
    self.content_ = content
  def isleaf(self):
    assert(self.content_)
    return isinstance(self.content_, str)
  def iscomposer(self):
    assert(self.content_)
    return isinstance(self.content_, list)
  def treesize(self):
    assert(self.content_)
    if self.isleaf():
      return 1
    elif self.iscomposer():
      num = 1
      for each in self.content_:
        num = num + each.treesize()
      return num
  def _dfsplrdump(self, tl, tr):
    assert(self.content_)
    if self.isleaf():
      tl.append(self)
    elif self.iscomposer():
      tr.insert(0, self)
      for each in self.content_:
        each._dfsplrdump(tl, tr)
  def dfsplrdump(self):
    tl = []
    tr = []
    # depth first search + parent left right order
    self._dfsplrdump(tl, tr)
    traj = tl+tr
    # trace back for the real trajectory
    trajectory = []
    for i in range(self.treesize()):
      trajectory.append(None)
    for i in list(range(len(traj))):
      tnode = traj[i]
      if i == len(traj)-1: # root node
        trajectory[i] = -1
      else: # normal node
        # search his father
        for j in list(range(i, len(traj))):
          if traj[j].iscomposer(): # father must be composer
            if tnode in traj[j].content_:
              trajectory[i] = j
    # index correction
    trajectory = list(map(lambda x: x+1, trajectory))
    return trajectory
  def sosdump(self):
    words = []
    self._sosdump(words)
    return words
  def _sosdump(self, words):
    if self.isleaf():
      words.append(self.content_)
    elif self.iscomposer():
      for each in self.content_:
        each._sosdump(words)

def unit_Tnode():
  log.debug('unit test: Tnode')
  a = Tnode('ROOT', 'hello')
  a.dump()
  print(a.sosdump())
  assert(a.treesize() == 1)
  a.dfsplrdump()
  a = Tnode('ROOT', [ Tnode('NN', 'nn1'), Tnode('NP', 'np2'),
    Tnode('XX', [ Tnode('x1', 'x1'), Tnode('x2', 'x2') ]) ])
  a.dump()
  print(a.sosdump())
  a.dfsplrdump()
  assert(a.treesize() == 6)
  a = None
  log.debug('unit test: OK')
if debug: unit_Tnode()

if debug: log.debug('init seeknext()')
def seeknext(cursor, raw):
  if cursor >= len(raw)-1:
    return len(raw)-1
  while raw[cursor].isspace() or raw[cursor] == '\n': # eat ' ' and '\n'
    cursor = cursor + 1
  return cursor

def unit_seeknext():
  log.debug('unit test: seeknext()')
  msg = '0ello   1eek            2eek 3eek'
  cur = 0
  print(msg[cur])
  cur = 5
  cur = seeknext(cur, msg)
  print(msg[cur])
  cur = 12
  cur = seeknext(cur, msg)
  print(msg[cur])
  cur = 28
  cur = seeknext(cur, msg)
  print(msg[cur])
  msg = None
  cur = None
  log.debug('unit test: OK')
if debug: unit_seeknext()

if debug: log.debug('init lex()')
def lex(raw):
  '''
  input: raw text
  output: lex parsed list
  '''
  assert(len(raw) > 0)
  cur = 0
  lexout = []
  while cur < len(raw):
    cur = seeknext(cur, raw)
    token = ''
    if raw[cur].isalnum() or raw[cur]=='$' or raw[cur]=='-':
      while raw[cur].isalnum() or raw[cur]=='$' or raw[cur]=='-':
        token = token+raw[cur]
        cur = cur + 1
    elif raw[cur] in [ ')', '(', '.' ]:
      token = raw[cur]
      cur = cur + 1
    else:
      char = raw[cur]
      if char in [ '\n' ]:
        cur = cur + 1
        continue
      else:
        print('what is [%s][%s]?'%(bytes(char.encode()), char[0]))
        #cur = cur + 1
        #cur = seeknext(cur, raw)
        #token = char
        assert(None)
    lexout.append(token)
  return lexout

def unit_lex():
  log.debug('unit test: lex()')
  msg = '(ROOT hello)'
  print(lex(msg))
  msg = '(ROOT (DT a) (NP cat))'
  print(lex(msg))
  msg = '(ROOT (DT a) (NP cat) )'
  print(lex(msg))
  log.debug('unit test: OK')
if debug: unit_lex()

if debug: log.debug('init yyparse()')
def yyparse(lexout, cur=0):
  '''
  input: lexout and optionally a cursor
  output: Tnode instance, and new cursor
  '''
  assert(lexout[cur] == '(')
  cur = cur + 1
  attr = lexout[cur]
  cur = cur + 1
  content = None
  if lexout[cur] != '(': # leaf node
    content = lexout[cur]
    cur = cur + 1 # cur> )
    if lexout[cur] == ')': # ok
      cur = cur + 1 # cur> next
    else:
      print('syntax error')
      return None
  elif lexout[cur] == '(': # composer node
    content = []
    while lexout[cur] == '(':
      yyout = yyparse(lexout, cur)
      tnode = yyout[0]
      cur   = yyout[1]
      content.append(tnode)
    if lexout[cur] == ')': # ok
      cur = cur + 1
    else:
      print('syntax error')
      return None
  else:
    print('syntax error')
    return None
  return (Tnode(attr, content), cur)

def unit_yyparse():
  log.debug('unit test: yyparse()')
  lexout = lex('(ROOT hello)')
  tree = yyparse(lexout)[0]
  tree.dump()
  lexout = lex('(ROOT (DT a) (NP cat))')
  tree = yyparse(lexout)[0]
  tree.dump()
  lexout = lex('(ROOT (NP (DT a) (NP cat)) (VP sits) (XX on) (NP (YY the) (NP mat)))')
  tree = yyparse(lexout)[0]
  tree.dump()
  lexout = lex('''
(ROOT
  (NP
    (NP (DT a) (NN person) (NN standing))
    (PP (IN inside))
    (PP (IN of)
      (NP (DT a) (NN phone) (NN booth)))
    (. .)))''')
  tree = yyparse(lexout)[0]
  tree.dump()
  lexout = None
  tree = None
  log.debug('unit test: OK')
if debug: unit_yyparse()

def unit_coco_tree():
  # test algorithm with the tree from coco
  tree_raw = '''
(ROOT
  (NP
    (NP (DT a) (NN person) (NN standing))
    (PP (IN inside))
    (PP (IN of)
      (NP (DT a) (NN phone) (NN booth)))
    (. .)))
'''
  lexout = lex(tree_raw)
  print(lexout)
  tree = yyparse(lexout)[0]
  tree.dfsplrdump()

if debug: 
  log.debug('unit test: coco tree test')
  unit_coco_tree()

log.debug('check argv')
if len(sys.argv) != 2:
  print('illegal argument list')
  exit (1)

log.warn('start parsing')

log.info('read specified file')
with open(sys.argv[1], 'r') as f:
  trees_raw = f.read()
log.debug('file size %d'% len(trees_raw))

log.info('dump file chars')
chars = []
for char in trees_raw:
  if not char in chars:
    chars.append(char)
print(chars)

log.info('lexical analysis')
lexout = lex(trees_raw)
log.debug('lexical output length %d'% len(lexout))

log.info('perform yyparse')
cur = 0
trees = []
#yyparse(lexout)[0].dump()
yyout = yyparse(lexout, cur)
trees.append(yyout[0])
cur   = yyout[1]
print(len(trees))
while cur < len(lexout)-1:
  yyout = yyparse(lexout, cur)
  tnode = yyout[0]
  cur   = yyout[1]
  trees.append(tnode)
  print(len(trees))
log.debug('parsed %d trees'% len(trees))

if debug:
  log.info('dump parsed trees')
  for each in trees:
    each.dump()

log.info('dump parent pointer tree of parsed trees')
ppts = []
for each in trees:
  ppt = each.dfsplrdump()
  print(ppt)
  ppts.append(map(str, ppt))

log.info('write result into %s'%(sys.argv[1]+'.ppts'))
pptst = ''
for each in ppts:
  pptst = pptst + '|'.join(each) + '\n'
print(pptst, end='')
with open(sys.argv[1]+'.ppts', 'w+') as f:
  f.write(pptst)

log.info('dump sostree and write them to %s'%(sys.argv[1]+'.soss'))
soss = ''
for each in trees:
  soss = soss + '|'.join(each.sosdump()) + '\n'
print(soss, end='')
with open(sys.argv[1]+'.soss', 'w+') as f:
  f.write(soss)

log.warn('parse complete')
