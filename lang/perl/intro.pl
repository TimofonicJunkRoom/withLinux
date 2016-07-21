#!/usr/bin/perl
use strict;
use warnings;

# basic syntax
print "Perl quick Introduction\n";
print "Reference: http://perldoc.perl.org/perlintro.html\n";
print 'in this case \n is not excaped';
print "\n";
print 42, "\n";
print ("hello perl, you can omit parentheses.\n");

# variables
print " -- [scalar variable] --\n";
my $animal = "camel";
my $answer = 42;
print $animal;
print "The animal is $animal\n";
print "The square of $answer is ", $answer * $answer, "\n";

print " -- [array] --\n";
my @animals = ("camel", "llama", "owl");
my @numbers = (23, 42, 69);
my @mixed   = ("camel", 42, 1.23);
print $animals[0];              # prints "camel"
print $animals[1];              # prints "llama"
print $mixed[$#mixed];       # last element, prints 1.23
print "\n";
if (@animals < 5) {
  print "array @animals < 5\n";
}

print "TODO\\nn";
