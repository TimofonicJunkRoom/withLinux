#!/usr/bin/python3
# @ref https://wiki.python.org/moin/PyOpenGL

import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

def drawFunc():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glRotatef(0.1, 1, 1, 1)
    glutWireTeapot(0.5)
    glFlush()

glutInit(sys.argv)
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH)
glutInitWindowSize(800, 600)
glutCreateWindow("Demo")
glutDisplayFunc(drawFunc)
glutIdleFunc(drawFunc)
glutMainLoop()
