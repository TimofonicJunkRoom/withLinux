#!/usr/bin/python3
# reference: pygame tutorial, including chimp
# reference: pygame examples
# reference: pygame/docs/tut/DisplayModes.html
# reference: Beginning Game Development with Python and Pygame –From Novice to Professional
'''
pygame.display.set_mode((width, height), flags, depth)
'''

import os
import sys

import pygame as pg
from pygame.locals import *
assert(pg.font)

arrow = ( "xX                      ",
          "X.X                     ",
          "X..X                    ",
          "X...X                   ",
          "X....X                  ",
          "X.....X                 ",
          "X......X                ",
          "X.......X               ",
          "X........X              ",
          "X.........X             ",
          "X......XXXXX            ",
          "X...X..X                ",
          "X..XX..X                ",
          "X.X XX..X               ",
          "XX   X..X               ",
          "X     X..X              ",
          "      X..X              ",
          "       X..X             ",
          "       X..X             ",
          "        XX              ",
          "                        ",
          "                        ",
          "                        ",
          "                        ")

def load_image(name, colorkey=None):
  fullname = os.path.join('data',name)
  try:
    image = pg.image.load(fullname)
  except (pg.error, message):
    print('E: missing image', fullname)
    raise (SystemExit, message)
  image = image.convert()
  if colorkey is not None:
    if colorkey is -1:
      colorkey = image.get_at((0,0))
    image.set_colorkey(colorkey, RLEACCEL)
  return image, image.get_rect()

class Fist(pg.sprite.Sprite):
  ''' following mouse '''
  def __init__(self):
    pg.sprite.Sprite.__init__(self)
    self.image, self.rect = load_image('asprite.bmp',-1)
  def update(self):
    pos = pg.mouse.get_pos()
    self.rect.midtop = pos

class Ball(pg.sprite.Sprite):
  ''' auto moves '''
  def __init__(self):
    pg.sprite.Sprite.__init__(self)
    self.image, self.rect = load_image('asprite.bmp',-1)
    screen = pg.display.get_surface()
    self.area = screen.get_rect()
    self.speed = [5,5]
  def update(self):
    newpos = self.rect.move(self.speed)
    if not self.area.contains(newpos):
      if self.rect.left < self.area.left or \
         self.rect.right > self.area.right:
        self.speed[0] = -self.speed[0]
      if self.rect.top > self.area.top or \
         self.rect.bottom < self.area.bottom:
        self.speed[1] = -self.speed[1]
      newpos = self.rect.move(self.speed)
    self.rect = newpos
      
class Chimp(pg.sprite.Sprite):
  ''' auto moves '''
  def __init__(self):
    pg.sprite.Sprite.__init__(self)
    self.image, self.rect = load_image('asprite.bmp',-1)
    screen = pg.display.get_surface()
    self.area = screen.get_rect()
    self.rect.topleft = 100, 100
    self.move = 9
    self.dizzy = 0
  def update(self):
    newpos = self.rect.move((self.move, 0))
    if not self.area.contains(newpos):
      if self.rect.left < self.area.left or \
         self.rect.right > self.area.right:
        self.move = - self.move
        newpos = self.rect.move((self.move, 0))
        self.image = pg.transform.flip(self.image,1,0)
    self.rect = newpos

def main():
  print('my pygame ...')
  pg.init()
  pg.font.init()
  wsize = width, height = 1280, 720
  os.environ['SDL_VIDEO_CENTERED'] = '1'
  # flags: NOFRAME, FULLSCREEN
  screen = pg.display.set_mode(wsize, NOFRAME, 0)
  pg.display.set_caption('my Pygame')
  #pg.mouse.set_visible(0)

  print('background')
  background = pg.Surface(screen.get_size())
  background = background.convert()
  background.fill((0,0,0))

  print('font')
  font = pg.font.Font(None, 36)
  text = font.render("Hello !", 1, (123,123,123))
  textpos = text.get_rect(centerx=background.get_width()/2)
  background.blit(text, textpos)

  screen.blit(background, (0,0))
  pg.display.flip()

  print('objects')
  ball = Ball()
  fist = Fist()
  chimp = Chimp()
  allsprites = pg.sprite.RenderPlain((fist,chimp,ball))
  clock = pg.time.Clock()

  print('set mouse')
  hotspot = None
  for y in range(len(arrow)):
    for x in range(len(arrow)):
      if arrow[y][x] in ['x', ',', 'O']:
        hotspot = x, y
        break
    if hotspot != None:
      break
  assert(hotspot)
  s2 = []
  for line in arrow:
    s2.append(line.replace('x','X').replace(',','.').replace('O','o'))
  cursor, mask = pg.cursors.compile(s2, 'X', '.', 'o')
  cursize = len(arrow[0]), len(arrow)
  pg.mouse.set_cursor(cursize, hotspot, cursor, mask)

  print('main loop ...')
  running = True
  while running:
    clock.tick(60)
    # event
    for event in pg.event.get():
      if event.type == pg.QUIT:
        running = False
      if event.type == KEYDOWN:
        if event.key == K_ESCAPE:
          print('caught ESC, stopping ...')
          running = False
    # update sprites and screen
    allsprites.update()
    screen.blit(background, (0,0))
    allsprites.draw(screen)
    pg.display.flip()
  # end while
  quit()

if __name__ == '__main__':
  main()
