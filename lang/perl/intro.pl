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
print @animals[0,1], "\n";
print @animals[0..2], "\n";
print @animals[1..$#animals], "\n"; # slicing operation

my @sorted = sort @animals;
my @backwards = reverse @numbers;
print @sorted, "\n";
print @backwards, "\n";

my %fruit_color_simple = ("apple", "red", "banana", "yellow"); # hashes
my %fruit_color = (
  apple => "red",
  banana => "yellow",
);
print %fruit_color, "\n";
print $fruit_color{"apple"}, "\n";
my @fruits = keys %fruit_color; # get keys of the hash table
my @colors = values %fruit_color; # get values from the hash table

# variable scoping
## my $var = "value"; # create a local variable
## $var = "value";    # create a global variable

# conditional and loop constructs

##if ( condition ) {
##  ...
##} elsif ( condition2 ) {
##  ...
##} else {
##  ...
##}

##unless ( condition ) {
##  ... 
##}
##equivalant to 
##if (! condition ) { ... }

## if ($zippy) {
##     print "Yow!"; // traditional way
## }
## print "Yow!" if $zippy; // perlish
## print "We have no bananas" unless $bananas; // perlish

## while (condition) { ... }
## until (condition) { ... }
## print "hello" while 1;

print "TODO\n";
