#!/usr/bin/python3
'''
unittest
https://docs.python.org/3/library/unittest.html?highlight=unittest
'''

import unittest

class TestStringMethods(unittest.TestCase):
  def test_upper(self):
    self.assertEqual('foo'.upper(), 'FOO')
  def test_isupper(self):
    self.assertTrue('FOO'.isupper())
    self.assertFalse('Foo'.isupper())
  def test_split(self):
    s = 'fiat lux'
    self.assertEqual(s.split(), ['fiat', 'lux'])
    with self.assertRaises(TypeError):
      s.split(2)

if __name__ == '__main__':
  print('''
append -v to the argument list to toggle verbose flag

you can run the unittest in this way:
 $ python3 -m unittest [-v] [script]
''')
  unittest.main()
